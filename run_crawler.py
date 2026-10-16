import os
import json
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

# ==================== إعدادات البيئة ====================
HF_REPO_ID = os.getenv("HF_REPO_ID", "Moo408/dat-9f82c4")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_DIR = "data_sync"
CRAWLER_DIR = "crawler_output"

os.makedirs(LOCAL_DIR, exist_ok=True)
api = HfApi(token=HF_TOKEN)

def extract_year_safely(date_val):
    """استخراج السنة بأمان مع دعم القيم الفارغة والتالفة"""
    if pd.isna(date_val) or not str(date_val).strip():
        return "unknown"
    s = str(date_val).strip()
    try:
        # يدعم صيغ ISO مثل 2026-10-16T12:00:00
        if len(s) >= 4 and s[:4].isdigit():
            year_num = int(s[:4])
            if 2015 <= year_num <= 2035:
                return str(year_num)
        # محاولة التحويل عبر pandas
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return str(dt.year)
    except Exception:
        pass
    return "unknown"

def sync_channels():
    """دمج وتحديث ملف القنوات c.parquet"""
    print("\n[*] --- Processing Channels (c.parquet) ---")
    new_c_path = os.path.join(CRAWLER_DIR, "c.parquet")
    if not os.path.exists(new_c_path):
        print("[-] No new channels crawled this round.")
        return

    new_c_df = pd.read_parquet(new_c_path)
    print(f"[+] Loaded {len(new_c_df):,} newly scraped/discovered channels.")

    existing_c_df = pd.DataFrame()
    try:
        print("[*] Downloading existing c.parquet from Hugging Face...")
        old_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="c.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        existing_c_df = pd.read_parquet(old_path)
        print(f"[+] Loaded existing c.parquet with {len(existing_c_df):,} channels.")
    except Exception as e:
        print(f"[-] Note: Starting c.parquet fresh ({e})")

    # دمج وتحديث القنوات
    if not existing_c_df.empty:
        merged_c = pd.concat([existing_c_df, new_c_df], ignore_index=True)
    else:
        merged_c = new_c_df

    # إزالة التكرار بالـ username مع الاحتفاظ بآخر تحديث
    merged_c = merged_c.drop_duplicates(subset=["username"], keep="last")
    out_c_file = os.path.join(LOCAL_DIR, "c.parquet")
    merged_c.to_parquet(out_c_file, compression="zstd", index=False)
    print(f"[✓] Saved updated c.parquet with {len(merged_c):,} channels.")

    print("[*] Uploading c.parquet to Hugging Face...")
    api.upload_file(
        path_or_fileobj=out_c_file,
        path_in_repo="c.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Auto-update channels: {len(merged_c):,} total"
    )
    print("[✓] c.parquet successfully uploaded!")

def sync_partitioned_posts():
    """تقسيم ودمج المنشورات سنوياً مع التعامل مع التواريخ غير المحددة"""
    print("\n[*] --- Processing Partitioned Posts ---")
    new_p_path = os.path.join(CRAWLER_DIR, "p.parquet")
    
    # 1. فحص هل هناك بوستات جديدة من الكراولر
    new_posts_df = pd.DataFrame()
    if os.path.exists(new_p_path):
        new_posts_df = pd.read_parquet(new_p_path)
        print(f"[+] Loaded {len(new_posts_df):,} new posts from crawler.")
    else:
        print("[-] No new posts crawled this round.")

    # 2. فحص إذا كان ملف p.parquet القديم الموحد موجوداً لنقله إلى النظام المقسم لأول مرة
    legacy_old_p_df = pd.DataFrame()
    try:
        legacy_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="p.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        legacy_old_p_df = pd.read_parquet(legacy_path)
        print(f"[!] Found legacy single p.parquet with {len(legacy_old_p_df):,} posts. Migrating to partitions...")
    except Exception:
        pass

    # تجميع كل الداتا الجديدة/المرحلة
    combined_incoming = pd.concat([legacy_old_p_df, new_posts_df], ignore_index=True)
    if combined_incoming.empty:
        print("[-] Nothing to sync for posts.")
        return

    # استخراج السنة وتصنيف البوستات
    combined_incoming["year_partition"] = combined_incoming["date"].apply(extract_year_safely)
    target_years = combined_incoming["year_partition"].unique()

    manifest_data = {}

    for year in target_years:
        partition_filename = f"posts_{year}.parquet"
        chunk_new = combined_incoming[combined_incoming["year_partition"] == year].drop(columns=["year_partition"])

        # تنزيل الملف الخاص بهذه السنة إن وُجد مسبقاً في Hugging Face
        existing_year_df = pd.DataFrame()
        try:
            year_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=partition_filename,
                repo_type="dataset",
                token=HF_TOKEN
            )
            existing_year_df = pd.read_parquet(year_path)
        except Exception:
            pass

        # دمج واستبعاد التكرار
        if not existing_year_df.empty:
            merged_year = pd.concat([existing_year_df, chunk_new], ignore_index=True)
        else:
            merged_year = chunk_new

        # إزالة التكرار مع الفرز التنازلي
        if "id" in merged_year.columns:
            merged_year = merged_year.drop_duplicates(subset=["id"], keep="last")
        else:
            merged_year = merged_year.drop_duplicates(subset=["username", "post_id"], keep="last")

        if "date" in merged_year.columns:
            merged_year = merged_year.sort_values(by="date", ascending=False)

        # حفظ الملف محلياً
        local_year_file = os.path.join(LOCAL_DIR, partition_filename)
        merged_year.to_parquet(local_year_file, compression="zstd", index=False)
        print(f"[+] Partition '{partition_filename}' prepared: {len(merged_year):,} posts.")

        # رفع الملف للسحابة
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
            "updated_at": datetime.utcnow().isoformat()
        }

    # رفع ملف manifest.json لتسهيل القراءة على المتصفح
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
    print("[✓] Partitions and Manifest successfully synced to Hugging Face!")

def main():
    print("=" * 60)
    print("🚀 Running Partitioned Pipeline Sync Engine")
    print("=" * 60)
    sync_channels()
    sync_partitioned_posts()
    print("=" * 60)
    print("[✓] All sync tasks finished successfully!")

if __name__ == "__main__":
    main()
