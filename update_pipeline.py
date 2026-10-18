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

def get_curated_keywords():
    """قراءة قائمة الكلمات المفتاحية للاهتمامات من ملف JSON"""
    kw_file = "interests.json"
    if os.path.exists(kw_file):
        try:
            with open(kw_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                words = data.get("keywords", [])
                return [w.lower().strip() for w in words