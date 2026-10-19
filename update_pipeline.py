import os
import re
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

# قائمة الحظر الصارم والفوري لإعلانات القمار والسبام الهندي والمتاجر
HARD_SPAM_BLOCKLIST = [
    "₹", "yono", "rummy", "bhailog", "bhai", "loot", "win cash",
    "t.me/+", "t.me/joinchat", "bonus upto", "free paytm", "1xbet",
    "betwinner", "melbet", "casino", "poker", "addlist/"
]

def get_cloudflare_blacklist():
    """جلب قائمة القنوات المحظورة مركزياً من Cloudflare KV لتطهيرها نهائياً"""
    cf_token = os.getenv("CF_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_kv_id = os.getenv("CF_KV_NAMESPACE_ID")

    if not (cf_token and cf_account and cf_kv_id):
        return set()

    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/blacklist"
    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json"
    }

    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            bl_list = res.json()
            if isinstance(bl_list, list):
                blacklist_set = set(str(u).lower().strip() for u in bl_list if str(u).strip())
                print(f"[*] Retrieved {len(blacklist_set)} blacklisted channels from Cloudflare KV for permanent purge.")
                return blacklist_set
    except Exception as e:
        print(f"[-] Could not fetch blacklist from Cloudflare: {e}")

    return set()

def load_user_keywords():
    """قراءة الكلمات المفتاحية من ملف interests.json حصراً دون أي إضافات خارجية"""
    kw_file = "interests.json"
    if not os.path.exists(kw_file):
        print("[!] WARNING: interests.json file NOT FOUND in root directory!")
        return []

    try:
        with open(kw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_words = data.get("keywords", [])
            keywords = [str(w).lower().strip() for w in raw_words if str(w).strip()]
            print(f"[✓] Successfully loaded {len(keywords)} EXACT keywords from interests.json.")
            return keywords
    except Exception as e:
        print(f"[-] Error parsing interests.json: {e}")
        return []

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

def sync_channels(blacklist):
    new_c_path = os.path.join(CRAWLER_DIR, "c.parquet")
    new_c_df = pd.DataFrame()
    if os.path.exists(new_c_path):
        new_c_df = pd.read_parquet(new_c_path)

    existing_c_df = pd.DataFrame()
    try:
        old_path = hf_hub_download(repo_id=HF_REPO_ID, filename="c.parquet", repo_type="dataset", token=HF_TOKEN)
        existing_c_df = pd.read_parquet(old_path)
    except Exception:
        pass

    merged_c = pd.concat([existing_c_df, new_c_df], ignore_index=True)
    if merged_c.empty:
        return 0, 0, 0

    merged_c = merged_c.drop_duplicates(subset=["username"], keep="last")

    # الحذف الفيزيائي الفعلي للقنوات المحظورة من قاعدة البيانات
    if blacklist and "username" in merged_c.columns:
        initial_c_count = len(merged_c)
        merged_c = merged_c[~merged_c["username"].astype(str).str.lower().isin(blacklist)]
        purged_c = initial_c_count - len(merged_c)
        if purged_c > 0:
            print(f"[🔥] Permanently purged {purged_c:,} channels from c.parquet!")

    out_c_file = os.path.join(LOCAL_DIR, "c.parquet")
    merged_c.to_parquet(out_c_file, compression="zstd", index=False)

    api.upload_file(
        path_or_fileobj=out_c_file,
        path_in_repo="c.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Sync & purge channels: {len(merged_c):,} total"
    )

    total = len(merged_c)
    checked = merged_c["last_checked"].dropna().astype(str).str.strip().ne("").sum()
    remaining = total - checked
    return len(new_c_df), total, remaining

def push_to_cloudflare_kv(posts_df):
    """فلترة فورية فائقة السرعة تقتصر على أحدث المنشورات وتتم في ثوانٍ معدودة"""
    cf_token = os.getenv("CF_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_kv_id = os.getenv("CF_KV_NAMESPACE_ID")

    if not (cf_token and cf_account and cf_kv_id):
        print("[-] Skipping Cloudflare KV push: Secrets missing.")
        return

    keywords = load_user_keywords()
    if not keywords:
        print("[-] No keywords loaded from interests.json.")
        return

    print(f"[*] Optimizing push pipeline for {len(posts_df):,} posts...")

    # 1. الترتيب بالأحدث أولاً
    if "date" in posts_df.columns:
        posts_df = posts_df.sort_values(by="date", ascending=False)

    # 2. اقتطاع شريحة أحدث 15,000 منشور لفحصها بدل فحص المليون منشور بالكامل
    recent_candidates = posts_df.head(15000).copy()

    # 3. دمج كل كلمات الاهتمامات في نمط Regex واحد عملاق وفائق السرعة
    escaped_keywords = [re.escape(k) for k in keywords if k]
    combined_pattern_str = r'(?<![\w\u0621-\u064A])(' + '|'.join(escaped_keywords) + r')(?![\w\u0621-\u064A])'
    master_regex = re.compile(combined_pattern_str, re.IGNORECASE)

    # 4. دمج كلمات السبام في نمط واحد
    escaped_spam = [re.escape(s) for s in HARD_SPAM_BLOCKLIST if s]
    spam_pattern = re.compile('|'.join(escaped_spam), re.IGNORECASE)
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

    # 5. الفلترة السريعة المبنية على السلسلة النصية (Vectorized C-Engine)
    texts = recent_candidates["text"].fillna("").astype(str).str.lower()
    usernames = recent_candidates["username"].fillna("").astype(str).str.lower()

    valid_mask = texts.str.strip().ne("")
    no_spam_mask = ~texts.str.contains(spam_pattern, regex=True) & ~usernames.str.contains(spam_pattern, regex=True)
    no_chinese_mask = ~texts.str.contains(chinese_pattern, regex=True)
    tech_match_mask = texts.str.contains(master_regex, regex=True)

    final_mask = valid_mask & no_spam_mask & no_chinese_mask & tech_match_mask
    curated_df = recent_candidates[final_mask]

    print(f"[+] Instantly curated {len(curated_df):,} high-quality posts in ~1 second!")

    cols_to_keep = ["id", "username", "post_id", "text", "image", "video", "media_type", "views", "date"]
    available_cols = [c for c in cols_to_keep if c in curated_df.columns]

    latest_subset = curated_df[available_cols].head(3000).fillna("").to_dict(orient="records")

    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/latest_posts"
    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json"
    }

    res = requests.put(url, headers=headers, json=latest_subset)
    if res.status_code == 200:
        print(f"[✓] Successfully pushed {len(latest_subset):,} clean posts to Cloudflare KV!")
    else:
        print(f"[-] Cloudflare KV sync failed: {res.text}")

def sync_partitioned_posts(blacklist):
    new_p_path = os.path.join(CRAWLER_DIR, "p.parquet")
    new_posts_df = pd.DataFrame()
    if os.path.exists(new_p_path):
        new_posts_df = pd.read_parquet(new_p_path)

    manifest_data = {}
    try:
        m_path = hf_hub_download(repo_id=HF_REPO_ID, filename="manifest.json", repo_type="dataset", token=HF_TOKEN)
        with open(m_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception:
        manifest_data = {"2026": {"file": "posts_2026.parquet"}}

    if not new_posts_df.empty:
        new_posts_df["year_partition"] = new_posts_df["date"].apply(extract_year_safely)
        for y in new_posts_df["year_partition"].unique():
            if y not in manifest_data:
                manifest_data[y] = {"file": f"posts_{y}.parquet"}

    all_2026_df = pd.DataFrame()
    total_posts_all_years = 0

    for year, meta in manifest_data.items():
        partition_filename = meta.get("file", f"posts_{year}.parquet")
        existing_year_df = pd.DataFrame()
        try:
            year_path = hf_hub_download(repo_id=HF_REPO_ID, filename=partition_filename, repo_type="dataset", token=HF_TOKEN)
            existing_year_df = pd.read_parquet(year_path)
        except Exception:
            pass

        chunk_new = pd.DataFrame()
        if not new_posts_df.empty and "year_partition" in new_posts_df.columns:
            chunk_new = new_posts_df[new_posts_df["year_partition"] == year].drop(columns=["year_partition"], errors="ignore")

        merged_year = pd.concat([existing_year_df, chunk_new], ignore_index=True)
        if merged_year.empty:
            continue

        if "id" in merged_year.columns:
            merged_year = merged_year.drop_duplicates(subset=["id"], keep="last")
        else:
            merged_year = merged_year.drop_duplicates(subset=["username", "post_id"], keep="last")

        # الحذف الفيزيائي للبوستات التابعة لقنوات البلاك ليست
        if blacklist and "username" in merged_year.columns:
            initial_p_count = len(merged_year)
            merged_year = merged_year[~merged_year["username"].astype(str).str.lower().isin(blacklist)]
            purged_p = initial_p_count - len(merged_year)
            if purged_p > 0:
                print(f"[🔥] Permanently purged {purged_p:,} posts from {partition_filename}!")

        if year == "2026":
            all_2026_df = merged_year

        local_year_file = os.path.join(LOCAL_DIR, partition_filename)
        merged_year.to_parquet(local_year_file, compression="zstd", index=False)

        api.upload_file(
            path_or_fileobj=local_year_file,
            path_in_repo=partition_filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Sync & clean partition {partition_filename}: {len(merged_year):,} posts"
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

    # دفع المنشورات المفلترة الصافية لـ Cloudflare KV
    if not all_2026_df.empty:
        push_to_cloudflare_kv(all_2026_df)

    for y in manifest_data:
        total_posts_all_years += manifest_data[y].get("total_rows", 0)

    return len(new_posts_df), total_posts_all_years

def main():
    print("\n" + "=" * 65)
    print("🚀 بدء مزامنة ورفع وتطهير البيانات السحابية...")
    print("=" * 65)

    blacklist = get_cloudflare_blacklist()
    c_crawled, c_total, c_remaining = sync_channels(blacklist)
    p_crawled, p_total = sync_partitioned_posts(blacklist)

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
