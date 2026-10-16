import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download

# ==================== إعدادات البيئة ====================
HF_REPO_ID = os.getenv("HF_REPO_ID", "Moo408/dat-9f82c4")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_DIR = "data_sync"
CRAWLER_DIR = "crawler_output"

os.makedirs(LOCAL_DIR, exist_ok=True)
api = HfApi(token=HF_TOKEN)

def extract_year_safely(date_val):
    if pd.isna(date_val) or not str(date_val).strip():
        return "unknown"
    s = str(date_val).strip()
    try:
        if len(s) >= 4 and s[:4].isdigit():
            year_num = int(s[:4])
            if 2015 <= year_num <= 2035:
                return str(year_num)
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return str(dt.year)
    except Exception:
        pass
    return "unknown"

def sync_channels():
    new_c_path = os.path.join(CRAWLER_DIR, "c.parquet")
    if not os.path.exists(new_c_path):
        return 0, 0, 0

    new_c_df = pd.read_parquet(new_c_path)
    existing_c_df = pd.DataFrame()
    try:
        old_path = hf_hub_download(repo_id=HF_REPO_ID, filename="c.parquet", repo_type="dataset", token=HF_TOKEN)
        existing_c_df = pd.read_parquet(old_path)
    except Exception:
        pass

    merged_c = pd.concat([existing_c_df, new_c_df], ignore_index=True)
    merged_c = merged_c.drop_duplicates(subset=["username"], keep="last")

    out_c_file = os.path.join(LOCAL_DIR, "c.parquet")
    merged_c.to_parquet(out_c_file, compression="zstd", index=False)

    api.upload_file(
        path_or_fileobj=out_c_file,
        path_in_repo="c.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Sync channels: {len(merged_c):,} total"
    )

    total = len(merged_c)
    checked = merged_c["last_checked"].dropna().astype(str).str.strip().ne("").sum()
    remaining = total - checked
    return len(new_c_df), total, remaining

def push_to_cloudflare_kv(posts_df):
    """إرسال أحدث 3,000 منشور إلى Cloudflare KV لتصفح فوري وخفيف للموبايل"""
    cf_token = os.getenv("CF_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_kv_id = os.getenv("CF_KV_NAMESPACE_ID")

    if not (cf_token and cf_account and cf_kv_id):
        print("[-] Skipping Cloudflare KV push: Secrets missing.")
        return

    print("[*] Uploading top 3,000 latest posts to Cloudflare KV...")
    if "date" in posts_df.columns:
        posts_df = posts_df.sort_values(by="date", ascending=False)

    # اختيار الحقول الأساسية لتقليل حجم الـ JSON المرفوع
    cols_to_keep = ["id", "username", "post_id", "text", "image", "video", "media_type", "views", "date"]
    available_cols = [c for c in cols_to_keep if c in posts_df.columns]
    
    latest_subset = posts_df[available_cols].head(3000).fillna("").to_dict(orient="records")

    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/latest_posts"
    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json"
    }

    res = requests.put(url, headers=headers, json=latest_subset)
    if res.status_code == 200:
        print("[✓] Successfully synced latest posts to Cloudflare KV Edge!")
    else:
        print(f"[-] Cloudflare KV sync failed: {res.text}")

def sync_partitioned_posts():
    new_p_path = os.path.join(CRAWLER_DIR, "p.parquet")
    if not os.path.exists(new_p_path):
        return 0, 0

    new_posts_df = pd.read_parquet(new_p_path)
    new_posts_df["year_partition"] = new_posts_df["date"].apply(extract_year_safely)
    target_years = new_posts_df["year_partition"].unique()

    manifest_data = {}
    try:
        m_path = hf_hub_download(repo_id=HF_REPO_ID, filename="manifest.json", repo_type="dataset", token=HF_TOKEN)
        with open(m_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception:
        pass

    all_2026_df = pd.DataFrame()
    total_posts_all_years = 0

    for year in target_years:
        partition_filename = f"posts_{year}.parquet"
        chunk_new = new_posts_df[new_posts_df["year_partition"] == year].drop(columns=["year_partition"])

        existing_year_df = pd.DataFrame()
        try:
            year_path = hf_hub_download(repo_id=HF_REPO_ID, filename=partition_filename, repo_type="dataset", token=HF_TOKEN)
            existing_year_df = pd.read_parquet(year_path)
        except Exception:
            pass

        merged_year = pd.concat([existing_year_df, chunk_new], ignore_index=True)
        if "id" in merged_year.columns:
            merged_year = merged_year.drop_duplicates(subset=["id"], keep="last")
        else:
            merged_year = merged_year.drop_duplicates(subset=["username", "post_id"], keep="last")

        if year == "2026":
            all_2026_df = merged_year

        local_year_file = os.path.join(LOCAL_DIR, partition_filename)
        merged_year.to_parquet(local_year_file, compression="zstd", index=False)

        api.upload_file(
            path_or_fileobj=local_year_file,
            path_in_repo=partition_filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Sync partition {partition_filename}: {len(merged_year):,} posts"
        )

        manifest_data[year] = {
            "file": partition_filename,
            "total_rows": len(merged_year),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    manifest_path = os.path.join(LOCAL_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    api.upload_file(
        path_or_fileobj=manifest_path,
        path_in_repo="manifest.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="Update dataset partitions manifest"
    )

    # دفع أحدث المنشورات لـ Cloudflare KV
    if not all_2026_df.empty:
        push_to_cloudflare_kv(all_2026_df)

    for y in manifest_data:
        total_posts_all_years += manifest_data[y].get("total_rows", 0)

    return len(new_posts_df), total_posts_all_years

def main():
    c_crawled, c_total, c_remaining = sync_channels()
    p_crawled, p_total = sync_partitioned_posts()

    c_checked = c_total - c_remaining
    progress_pct = (c_checked / c_total * 100) if c_total > 0 else 0

    print("\n" + "╔" + "═" * 63 + "╗")
    print("║          🔥 تقرير إنجاز الجولة الحالية (Round Dashboard)       ║")
    print("╠" + "═" * 63 + "╣")
    print(f"║ 📥 قنوات تم فحصها وتحديثها الجولة دي : {c_crawled:,} قناة")
    print(f"║ 💬 منشورات جديدة تم إضافتها للسحابة   : +{p_crawled:,} بوست جديد")
    print("╠" + "═" * 63 + "╣")
    print(f"║ 📊 إجمالي القنوات الكلي في القاعدة    : {c_total:,} قناة")
    print(f"║ ✅ إجمالي القنوات المفحوصة حتى الآن  : {c_checked:,} قناة")
    print(f"║ ⏳ المتبقي في الطابور للفحص           : {c_remaining:,} قناة")
    print(f"║ 📈 نسبة إنجاز الطابور الكلي          : {progress_pct:.2f}%")
    print(f"║ 📚 إجمالي المنشورات الكلي في السحابة  : {p_total:,} بوست")
    print("╚" + "═" * 63 + "╝\n")

if __name__ == "__main__":
    main()
