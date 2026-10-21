import os
import re
import json
import requests
import pandas as pd
from collections import Counter
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download
from purge_rules import build_matcher, channels_to_purge, match_terms, normalize_username
from explorer_scoring import (AI_SCORE_THRESHOLD, build_model_patterns, load_model_terms, rank_items, rank_top_domains, score_ai_link)

# ==================== إعدادات البيئة ====================
HF_REPO_ID = os.getenv("HF_REPO_ID", "Moo408/dat-9f82c4")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_DIR = "data_sync"
CRAWLER_DIR = "crawler_output"

# سعة الخلاصة النشطة وفحص العينات الموسع
MAX_FEED_LIMIT = 8000
SAMPLE_SCAN_LIMIT = 40000

os.makedirs(LOCAL_DIR, exist_ok=True)
api = HfApi(token=HF_TOKEN)

# قائمة الحظر الصارم والفوري لإعلانات القمار والسبام الهندي والمتاجر
HARD_SPAM_BLOCKLIST = [
    "₹", "yono", "rummy", "bhailog", "bhai", "loot", "win cash",
    "t.me/+", "t.me/joinchat", "bonus upto", "free paytm", "1xbet",
    "betwinner", "melbet", "casino", "poker", "addlist/"
]

def cloudflare_blacklist_url():
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_kv_id = os.getenv("CF_KV_NAMESPACE_ID")
    if not (os.getenv("CF_API_TOKEN") and cf_account and cf_kv_id):
        return None
    return f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/blacklist"


def get_cloudflare_blacklist(required=False):
    """Load the persistent blacklist; fail closed when a safe sync requires it."""
    url = cloudflare_blacklist_url()
    cf_token = os.getenv("CF_API_TOKEN")
    if not url:
        if required:
            raise RuntimeError("Cloudflare KV credentials are missing; refusing to sync without the persistent blacklist")
        return set()

    headers = {"Authorization": f"Bearer {cf_token}", "Content-Type": "application/json"}
    try:
        res = requests.get(url, headers=headers, timeout=20)
        if res.status_code == 404:
            print("[*] No Cloudflare blacklist exists yet; starting with an empty list.")
            return set()
        res.raise_for_status()
        values = res.json()
        if not isinstance(values, list):
            raise ValueError("Cloudflare blacklist value must be a JSON array")
        blacklist = {normalize_username(value) for value in values if normalize_username(value)}
        print(f"[*] Retrieved {len(blacklist)} blacklisted channels from Cloudflare KV.")
        return blacklist
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not safely load Cloudflare blacklist: {exc}") from exc
        print(f"[-] Could not fetch Cloudflare blacklist: {exc}")
        return set()


def put_cloudflare_blacklist(blacklist):
    """Persist channel purges so Worker feeds and future sync runs keep them removed."""
    url = cloudflare_blacklist_url()
    if not url:
        raise RuntimeError("Cannot persist purged channels: Cloudflare KV credentials are missing")
    headers = {"Authorization": f"Bearer {os.getenv('CF_API_TOKEN')}", "Content-Type": "application/json"}
    response = requests.put(url, headers=headers, data=json.dumps(sorted(blacklist)), timeout=30)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError(f"Cloudflare rejected blacklist update: {result.get('errors', result)}")
    print(f"[✓] Persisted {len(blacklist)} blacklisted channels to Cloudflare KV.")


def load_purge_rules(path="purge_rules.json"):
    """Load exact whole-word/phrase channel purge rules from repository JSON."""
    with open(path, "r", encoding="utf-8") as rules_file:
        data = json.load(rules_file)
    if not isinstance(data, dict) or not isinstance(data.get("terms"), list):
        raise ValueError(f"{path} must contain a JSON object with a 'terms' array")
    rules = build_matcher(data["terms"])
    print(f"[*] Loaded {len(rules)} exact whole-word/phrase purge rules.")
    return rules

def load_user_keywords():
    """قراءة الكلمات المفتاحية من ملف interests.json حصراً"""
    kw_file = "interests.json"
    if not os.path.exists(kw_file):
        print("[!] WARNING: interests.json file NOT FOUND!")
        return []

    try:
        with open(kw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_words = data.get("keywords", [])
            keywords = [str(w).lower().strip() for w in raw_words if str(w).strip()]
            print(f"[✓] Loaded {len(keywords)} EXACT keywords from interests.json.")
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

    if blacklist and "username" in merged_c.columns:
        initial_c_count = len(merged_c)
        normalized_names = merged_c["username"].map(normalize_username)
        merged_c = merged_c[~normalized_names.isin(blacklist)]
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

def extract_and_push_explorer_data(posts_df):
    """رادار أذكى: تصنيف دقيق لأدوات الذكاء الاصطناعي + روابط آخر 100 مستخرجة"""
    cf_token = os.getenv("CF_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_kv_id = os.getenv("CF_KV_NAMESPACE_ID")

    if not (cf_token and cf_account and cf_kv_id):
        return

    print("[*] Running Smarter Multi-Asset Discovery...")
    sample_df = posts_df.copy()
    if "date" in sample_df.columns:
        try:
            sample_df = sample_df.sort_values(by="date", ascending=False)
        except Exception:
            pass
    sample_df = sample_df.head(SAMPLE_SCAN_LIMIT)

    model_terms = load_model_terms("ai_models_catalog.json")
    model_patterns = build_model_patterns(model_terms)
    print(f"[*] Loaded {len(model_terms)} AI model/vendor terms for classification.")

    ai_sites_counter = Counter()
    ai_sites_channels = {}
    ai_sites_meta = {}
    web_gems_counter = Counter()
    web_gems_channels = {}
    web_gems_meta = {}
    github_counter = Counter()
    github_channels = {}
    github_meta = {}
    all_domains_counter = Counter()
    all_domains_channels = {}
    all_domains_meta = {}
    tools_counter = Counter()

    url_pattern = re.compile(r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(/[^\s]*)?', re.IGNORECASE)
    github_pattern = re.compile(r'github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)', re.IGNORECASE)
    pkg_pattern = re.compile(r'(?:pip install|pkg install|apt install)\s+([a-zA-Z0-9_-]+)', re.IGNORECASE)

    EXCLUDED_DOMAINS = {
        "t.me", "telegram.org", "youtube.com", "youtu.be", "twitter.com", "x.com",
        "facebook.com", "instagram.com", "tiktok.com", "amzn.to", "amazon.com",
        "mediafire.com", "mega.nz", "drive.google.com", "play.google.com", "wa.me",
        "whatsapp.com", "tiny.cc", "bit.ly", "cutt.ly", "redd.it", "reddit.com",
        "aliexpress.com", "linkedin.com", "flipkart.com", "myntra.com", "temu.com",
        "shein.com", "google.com", "bing.com", "yahoo.com"
    }

    spam_pattern = re.compile('|'.join([re.escape(s) for s in HARD_SPAM_BLOCKLIST]), re.IGNORECASE)
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

    seen_links = set()


    for _, row in sample_df.iterrows():
        text = str(row.get("text", "") or "").strip()
        username = str(row.get("username", "") or "").strip().lower()
        post_id = str(row.get("post_id", "") or "").strip()
        date_val = str(row.get("date", "") or "").strip()
        if not text or text == "nan":
            continue

        text_lower = text.lower()
        if spam_pattern.search(text_lower) or spam_pattern.search(username) or chinese_pattern.search(text):
            continue

        for domain, path in url_pattern.findall(text):
            dom = domain.lower().replace("www.", "").strip()
            if dom in EXCLUDED_DOMAINS or "github.com" in dom:
                continue

            full_link = f"https://{dom}{path}" if path else f"https://{dom}"
            clean_link = full_link.split("?")[0].rstrip("/")
            if clean_link in seen_links:
                continue
            seen_links.add(clean_link)

            all_domains_counter[dom] += 1
            all_domains_channels.setdefault(dom, set()).add(username)
            if dom not in all_domains_meta:
                all_domains_meta[dom] = {
                    "url": clean_link,
                    "domain": dom,
                    "sample": text[:150],
                    "channel": username,
                    "date": date_val,
                }

            score = score_ai_link(dom, text_lower, model_patterns)
            if score >= AI_SCORE_THRESHOLD:
                ai_sites_counter[dom] += 1
                ai_sites_channels.setdefault(dom, set()).add(username)
                if dom not in ai_sites_meta:
                    ai_sites_meta[dom] = {"url": clean_link, "domain": dom, "sample": text[:150], "channel": username, "score": score}
            elif any(word in text_lower for word in ["api", "tool", "open source", "developer", "sdk", "cli", "framework", "library", "utility", "formatter", "converter", "editor", "generator", "checker", "validator", "parser", "viewer", "boilerplate", "documentation", "reference"]):
                web_gems_counter[dom] += 1
                web_gems_channels.setdefault(dom, set()).add(username)
                if dom not in web_gems_meta:
                    web_gems_meta[dom] = {"url": clean_link, "domain": dom, "sample": text[:150], "channel": username}

        for repo in github_pattern.findall(text):
            clean_repo = repo.strip().rstrip("/").rstrip(".")
            if len(clean_repo.split("/")) == 2 and not clean_repo.endswith((".git", ".png", ".jpg")):
                github_counter[clean_repo] += 1
                github_channels.setdefault(clean_repo, set()).add(username)
                if clean_repo not in github_meta:
                    github_meta[clean_repo] = {"name": clean_repo, "url": f"https://github.com/{clean_repo}", "sample": text[:150], "channel": username}

        for pkg in pkg_pattern.findall(text):
            clean_pkg = pkg.strip().lower()
            if 3 <= len(clean_pkg) <= 30:
                tools_counter[clean_pkg] += 1

    top_ai_tools = []
    for score, dom, count, unique_ch in rank_items(ai_sites_counter, ai_sites_channels)[:35]:
        meta = ai_sites_meta.get(dom, {})
        top_ai_tools.append({
            "title": dom,
            "url": meta.get("url", f"https://{dom}"),
            "mentions": count,
            "unique_channels": unique_ch,
            "score": meta.get("score", round(score, 2)),
            "preview": meta.get("sample", ""),
            "channel": meta.get("channel", "")
        })

    top_web_gems = []
    for score, dom, count, unique_ch in rank_items(web_gems_counter, web_gems_channels)[:35]:
        meta = web_gems_meta.get(dom, {})
        top_web_gems.append({
            "title": dom,
            "url": meta.get("url", f"https://{dom}"),
            "mentions": count,
            "unique_channels": unique_ch,
            "preview": meta.get("sample", ""),
            "channel": meta.get("channel", "")
        })

    top_github = []
    for score, repo, count, unique_ch in rank_items(github_counter, github_channels)[:35]:
        meta = github_meta.get(repo, {})
        top_github.append({
            "title": repo,
            "url": meta.get("url", f"https://github.com/{repo}"),
            "mentions": count,
            "unique_channels": unique_ch,
            "preview": meta.get("sample", ""),
            "channel": meta.get("channel", "")
        })

    top_100_domains = rank_top_domains(all_domains_counter, all_domains_channels)
    for item in top_100_domains:
        meta = all_domains_meta.get(item["domain"])
        if meta:
            item.update({
                "url": meta["url"],
                "sample": meta["sample"],
                "channel": meta["channel"],
                "date": meta["date"],
            })

    explorer_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "top_ai_tools": top_ai_tools,
        "top_web_gems": top_web_gems,
        "top_repos": top_github,
        "top_packages": [{"package": p, "mentions": c} for p, c in tools_counter.most_common(40)],
        "top_100_domains": top_100_domains
    }

    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/curated_explorer_data"
    headers = {"Authorization": f"Bearer {cf_token}", "Content-Type": "application/json"}
    res = requests.put(url, headers=headers, json=explorer_payload)
    if res.status_code == 200:
        print(f"[✓] Explorer Smart-Pushed: {len(top_ai_tools)} AI Tools, {len(top_web_gems)} Web Gems, {len(top_github)} Repos, {len(last_links)} recent links!")

def push_to_cloudflare_kv(posts_df):
    """فلترة وضخ حتى 8,000 منشور نشط مع ترتيب قائم على الجودة والتفاعل"""
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

    print(f"[*] Curating high-yield live feed from {len(posts_df):,} posts...")

    if "date" in posts_df.columns:
        posts_df = posts_df.sort_values(by="date", ascending=False)

    recent_candidates = posts_df.head(SAMPLE_SCAN_LIMIT).copy()

    escaped_keywords = [re.escape(k) for k in keywords if k]
    combined_pattern_str = r'(?<![\w\u0621-\u064A])(' + '|'.join(escaped_keywords) + r')(?![\w\u0621-\u064A])'
    master_regex = re.compile(combined_pattern_str, re.IGNORECASE)

    escaped_spam = [re.escape(s) for s in HARD_SPAM_BLOCKLIST if s]
    spam_pattern = re.compile('|'.join(escaped_spam), re.IGNORECASE)
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

    records = recent_candidates.to_dict(orient="records")
    clean_curated = []

    for r in records:
        text = str(r.get("text", "") or "").strip()
        username = str(r.get("username", "") or "").strip().lower()

        if not text or text.lower() == "nan":
            continue

        text_lower = text.lower()

        # طرد الرموز الصينية والسبام
        if chinese_pattern.search(text):
            continue
        if spam_pattern.search(text_lower) or spam_pattern.search(username):
            continue

        # مطابقة الكلمات التقنية الحصرية من interests.json
        if master_regex.search(text):
            clean_curated.append(r)
            if len(clean_curated) >= MAX_FEED_LIMIT:
                break

    print(f"[+] Successfully curated {len(clean_curated):,} clean tech posts (Cap: {MAX_FEED_LIMIT})!")

    cols_to_keep = ["id", "username", "post_id", "text", "image", "video", "media_type", "views", "date"]
    latest_subset = []
    for p in clean_curated:
        row_dict = {c: str(p.get(c, "") if pd.notna(p.get(c)) else "") for c in cols_to_keep}
        latest_subset.append(row_dict)

    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/latest_posts"
    headers = {"Authorization": f"Bearer {cf_token}", "Content-Type": "application/json"}
    res = requests.put(url, headers=headers, json=latest_subset)
    if res.status_code == 200:
        print(f"[✓] Live Feed pushed {len(latest_subset):,} posts to Cloudflare KV!")

    # تشغيل محرك استكشاف الكنوز والكيانات
    extract_and_push_explorer_data(posts_df)

def discover_purge_channels(manifest_data, new_posts_df, purge_rules, existing_manifest_years):
    """Scan every known history partition plus newly crawled posts before modifying data."""
    blocked = set()
    for year, meta in manifest_data.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid manifest entry for year {year}")
        filename = meta.get("file", f"posts_{year}.parquet")
        chunk = pd.DataFrame()
        if not new_posts_df.empty and "year_partition" in new_posts_df.columns:
            chunk = new_posts_df[new_posts_df["year_partition"] == year].drop(columns=["year_partition"], errors="ignore")
        existing = pd.DataFrame()
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, repo_type="dataset", token=HF_TOKEN)
            existing = pd.read_parquet(path)
        except Exception as exc:
            if year in existing_manifest_years:
                raise RuntimeError(f"Cannot safely inspect historical partition {filename}: {exc}") from exc
        partition = pd.concat([existing, chunk], ignore_index=True)
        if partition.empty:
            continue
        if "text" not in partition.columns:
            raise ValueError(f"Partition {filename} has no text column; refusing cleanup")
        records = partition[["username", "text"]].to_dict(orient="records") if "username" in partition.columns else [
            {"text": text} for text in partition["text"]
        ]
        blocked.update(channels_to_purge(records, purge_rules))
    return blocked


def sync_partitioned_posts(blacklist, purge_rules):
    new_p_path = os.path.join(CRAWLER_DIR, "p.parquet")
    new_posts_df = pd.DataFrame()
    if os.path.exists(new_p_path):
        new_posts_df = pd.read_parquet(new_p_path)

    manifest_loaded = False
    try:
        m_path = hf_hub_download(repo_id=HF_REPO_ID, filename="manifest.json", repo_type="dataset", token=HF_TOKEN)
        with open(m_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_loaded = True
    except Exception as exc:
        if purge_rules:
            raise RuntimeError(f"Cannot safely load full-history manifest: {exc}") from exc
        manifest_data = {"2026": {"file": "posts_2026.parquet"}}

    if not isinstance(manifest_data, dict):
        raise ValueError("Hugging Face manifest.json must be a JSON object")
    existing_manifest_years = set(manifest_data) if manifest_loaded else set()

    if not new_posts_df.empty:
        new_posts_df["year_partition"] = new_posts_df["date"].apply(extract_year_safely)
        for y in new_posts_df["year_partition"].unique():
            if y not in manifest_data:
                manifest_data[y] = {"file": f"posts_{y}.parquet"}

    if purge_rules:
        newly_blocked = discover_purge_channels(manifest_data, new_posts_df, purge_rules, existing_manifest_years) - blacklist
        if newly_blocked:
            blacklist.update(newly_blocked)
            put_cloudflare_blacklist(blacklist)
            print(f"[🔥] Exact rules matched and permanently blacklisted {len(newly_blocked):,} channels.")

    all_2026_df = pd.DataFrame()
    total_posts_all_years = 0

    for year, meta in manifest_data.items():
        partition_filename = meta.get("file", f"posts_{year}.parquet")
        existing_year_df = pd.DataFrame()
        try:
            year_path = hf_hub_download(repo_id=HF_REPO_ID, filename=partition_filename, repo_type="dataset", token=HF_TOKEN)
            existing_year_df = pd.read_parquet(year_path)
        except Exception as exc:
            if year in existing_manifest_years:
                raise RuntimeError(f"Cannot safely merge historical partition {partition_filename}: {exc}") from exc

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

        initial_p_count = len(merged_year)
        remove_rows = pd.Series(False, index=merged_year.index)
        normalized_names = pd.Series("", index=merged_year.index)
        if "username" in merged_year.columns:
            normalized_names = merged_year["username"].map(normalize_username)
            if blacklist:
                remove_rows |= normalized_names.isin(blacklist)
        if purge_rules and "text" in merged_year.columns:
            matched_rows = merged_year["text"].fillna("").map(lambda text: bool(match_terms(text, purge_rules)))
            # A match without an identifiable channel still removes the offending post.
            remove_rows |= matched_rows & normalized_names.eq("")
        merged_year = merged_year[~remove_rows]
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

    push_to_cloudflare_kv(all_2026_df)

    for y in manifest_data:
        total_posts_all_years += manifest_data[y].get("total_rows", 0)

    return len(new_posts_df), total_posts_all_years

def main():
    print("\n" + "=" * 65)
    print("🚀 بدء مزامنة ورفع وتوسيع استكشاف البيانات...")
    print("=" * 65)

    purge_rules = load_purge_rules()
    blacklist = get_cloudflare_blacklist(required=True)
    p_crawled, p_total = sync_partitioned_posts(blacklist, purge_rules)
    c_crawled, c_total, c_remaining = sync_channels(blacklist)

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
