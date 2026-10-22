import os
import re
import json
import math
import requests
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download
from purge_rules import build_matcher, channels_to_purge, match_terms, normalize_username
from explorer_scoring import (
    AI_SCORE_THRESHOLD, build_model_patterns, load_model_terms, load_category_terms,
    rank_items, rank_top_domains, score_ai_link, classify_category,
    calculate_trend_metrics, parse_utc_datetime, is_valid_github_repo,
)

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
    """Build categorized discoveries, publication-time trends, and measured source-post view velocity."""
    cf_token = os.getenv("CF_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_kv_id = os.getenv("CF_KV_NAMESPACE_ID")
    if not (cf_token and cf_account and cf_kv_id):
        return

    print("[*] Running categorized Discovery Hub analysis...")
    now_utc = datetime.now(timezone.utc)
    sample_df = posts_df.copy()
    if "date" in sample_df.columns:
        sample_df["_sort_date"] = pd.to_datetime(sample_df["date"], utc=True, errors="coerce")
        sample_df = sample_df.sort_values("_sort_date", ascending=False, na_position="last")
        sample_df = sample_df.drop(columns=["_sort_date"])
    sample_df = sample_df.head(SAMPLE_SCAN_LIMIT)

    model_terms = load_model_terms("ai_models_catalog.json")
    model_patterns = build_model_patterns(model_terms)
    category_terms = load_category_terms("ai_models_catalog.json")
    print(f"[*] Loaded {len(model_terms)} model/vendor terms and {len(category_terms)} category groups.")

    ai_counter = Counter()
    ai_channels = defaultdict(set)
    ai_meta = {}
    web_counter = Counter()
    web_channels = defaultdict(set)
    web_meta = {}
    github_counter = Counter()
    github_channels = defaultdict(set)
    github_meta = {}
    domain_counter = Counter()
    domain_channels = defaultdict(set)
    domain_meta = {}
    domain_occurrences = defaultdict(list)
    repo_occurrences = defaultdict(list)
    domain_urls = defaultdict(set)
    domain_category_votes = defaultdict(Counter)
    tools_counter = Counter()
    domain_view_delta = Counter()
    domain_view_rate = Counter()
    domain_view_samples = Counter()
    repo_view_delta = Counter()
    repo_view_rate = Counter()
    repo_view_samples = Counter()

    url_pattern = re.compile(r"""https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(/[^\s<>"']*)?""", re.IGNORECASE)
    github_pattern = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.IGNORECASE)
    pkg_pattern = re.compile(r"(?:pip install|pkg install|apt install)\s+([a-zA-Z0-9_.-]+)", re.IGNORECASE)

    excluded_domains = {
        "t.me", "telegram.org", "youtube.com", "youtu.be", "twitter.com", "x.com",
        "facebook.com", "instagram.com", "tiktok.com", "amzn.to", "amazon.com",
        "mediafire.com", "mega.nz", "drive.google.com", "play.google.com", "wa.me",
        "whatsapp.com", "tiny.cc", "bit.ly", "cutt.ly", "redd.it", "reddit.com",
        "aliexpress.com", "linkedin.com", "flipkart.com", "myntra.com", "temu.com",
        "shein.com", "google.com", "bing.com", "yahoo.com",
    }
    spam_pattern = re.compile("|".join(re.escape(s) for s in HARD_SPAM_BLOCKLIST), re.IGNORECASE)
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    utility_pattern = re.compile(
        r"\b(api|tool|open source|developer|sdk|cli|framework|library|utility|"
        r"formatter|converter|editor|generator|checker|validator|parser|viewer|"
        r"boilerplate|documentation|reference|self-hosted|docker|termux)\b",
        re.IGNORECASE,
    )

    def numeric(value):
        try:
            if pd.isna(value):
                return None
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    def view_metrics(key, is_repo=False):
        deltas = repo_view_delta if is_repo else domain_view_delta
        rates = repo_view_rate if is_repo else domain_view_rate
        samples = repo_view_samples if is_repo else domain_view_samples
        return {
            "source_post_views_delta": int(deltas.get(key, 0)) if samples.get(key, 0) else None,
            "source_post_views_per_hour": round(rates.get(key, 0.0), 2) if samples.get(key, 0) else None,
            "source_post_view_samples": int(samples.get(key, 0)),
        }

    def trend_for(key, occurrences, channels):
        result = calculate_trend_metrics(occurrences.get(key, []), channels.get(key, set()), now_utc)
        result["trend_status_label"] = {
            "new": "New", "rising": "Rising", "trending": "Trending",
            "popular": "Popular", "active": "Active",
            "insufficient_data": "Insufficient data",
        }.get(result["trend_status"], "Insufficient data")
        return result

    def category_for(domain, sample, votes=None):
        if votes:
            category, count = votes.most_common(1)[0]
            total = sum(votes.values())
            return {
                "category": category,
                "category_label": {
                    "ai_coding_agents": "AI Coding & Agents",
                    "local_ai": "Local AI & Models",
                    "creative_ai": "Image, Video & Audio AI",
                    "automation_api": "Automation, MCP & APIs",
                    "self_hosted_devtools": "Self-hosted & Developer Tools",
                    "ai_tools": "AI Tools",
                    "web_utilities": "Developer Utilities",
                }.get(category, "Developer Utilities"),
                "category_confidence": min(95, round(45 + 50 * count / max(total, 1))),
            }
        return classify_category(domain, sample, category_terms, model_patterns)

    rows_with_views = 0
    for _, row in sample_df.iterrows():
        post_text = str(row.get("text", "") or "").strip()
        username = str(row.get("username", "") or "").strip().lower().lstrip("@")
        post_id = str(row.get("post_id", "") or "").strip()
        date_val = str(row.get("date", "") or "").strip()
        post_dt = parse_utc_datetime(date_val)
        if not post_text or post_text.casefold() == "nan":
            continue
        text_lower = post_text.lower()
        if spam_pattern.search(text_lower) or spam_pattern.search(username) or chinese_pattern.search(post_text):
            continue

        # Count each domain/repository at most once per source post, not once per URL
        # and not once for the entire dataset. Repeated cross-channel posts still count.
        post_domain_links = defaultdict(set)
        post_domain_categories = {}
        post_domain_scores = {}
        for match in url_pattern.finditer(post_text):
            dom = match.group(1).lower().removeprefix("www.").strip(".")
            path = (match.group(2) or "").split("?")[0].split("#")[0].rstrip(".,;:!?)]}")
            if any(dom == excluded or dom.endswith("." + excluded) for excluded in excluded_domains):
                continue
            if dom == "github.com" or dom.endswith(".github.com"):
                continue
            clean_link = f"https://{dom}{path}".rstrip("/")
            post_domain_links[dom].add(clean_link)
            post_domain_categories[dom] = classify_category(dom, text_lower, category_terms, model_patterns)
            post_domain_scores[dom] = max(
                post_domain_scores.get(dom, 0),
                score_ai_link(dom, text_lower, model_patterns),
            )

        view_delta = numeric(row.get("views_delta"))
        view_rate = numeric(row.get("views_velocity_per_hour"))
        view_interval = numeric(row.get("views_interval_hours"))
        snapshot_at = parse_utc_datetime(row.get("scraped_at"))
        snapshot_age_hours = (
            (now_utc - snapshot_at).total_seconds() / 3600
            if snapshot_at else None
        )
        has_view_sample = (
            view_delta is not None and view_rate is not None
            and view_interval is not None and 0 < view_interval <= 24
            and snapshot_age_hours is not None
            and 0 <= snapshot_age_hours <= 24
        )
        if has_view_sample:
            rows_with_views += 1

        for dom, links in post_domain_links.items():
            domain_counter[dom] += 1
            if username:
                domain_channels[dom].add(username)
            domain_occurrences[dom].append((post_dt, username))
            domain_urls[dom].update(links)
            domain_category_votes[dom][post_domain_categories[dom]["category"]] += 1
            domain_meta.setdefault(dom, {
                "url": sorted(links)[0] if links else f"https://{dom}",
                "domain": dom,
                "sample": post_text[:180],
                "channel": username,
                "date": date_val,
            })
            score = post_domain_scores[dom]
            if score >= AI_SCORE_THRESHOLD:
                ai_counter[dom] += 1
                if username:
                    ai_channels[dom].add(username)
                ai_meta.setdefault(dom, {
                    "url": sorted(links)[0] if links else f"https://{dom}",
                    "domain": dom,
                    "sample": post_text[:180],
                    "channel": username,
                    "score": score,
                })
            elif utility_pattern.search(text_lower):
                web_counter[dom] += 1
                if username:
                    web_channels[dom].add(username)
                web_meta.setdefault(dom, {
                    "url": sorted(links)[0] if links else f"https://{dom}",
                    "domain": dom,
                    "sample": post_text[:180],
                    "channel": username,
                })
            if has_view_sample:
                domain_view_delta[dom] += int(view_delta)
                domain_view_rate[dom] += view_rate
                domain_view_samples[dom] += 1

        seen_repos_in_post = set()
        for repo in github_pattern.findall(post_text):
            clean_repo = repo.strip().rstrip("/").rstrip(".")
            if not is_valid_github_repo(clean_repo):
                continue
            normalized_repo = "/".join(part for part in clean_repo.split("/"))
            if normalized_repo.casefold() in seen_repos_in_post:
                continue
            seen_repos_in_post.add(normalized_repo.casefold())
            github_counter[normalized_repo] += 1
            if username:
                github_channels[normalized_repo].add(username)
            repo_occurrences[normalized_repo].append((post_dt, username))
            github_meta.setdefault(normalized_repo, {
                "name": normalized_repo,
                "url": f"https://github.com/{normalized_repo}",
                "sample": post_text[:180],
                "channel": username,
            })
            if has_view_sample:
                repo_view_delta[normalized_repo] += int(view_delta)
                repo_view_rate[normalized_repo] += view_rate
                repo_view_samples[normalized_repo] += 1

        seen_packages = set()
        for package in pkg_pattern.findall(post_text):
            clean_package = package.strip().lower().strip(".,;:!?)")
            if 3 <= len(clean_package) <= 40:
                seen_packages.add(clean_package)
        tools_counter.update(seen_packages)

    print(f"[*] Analyzed {len(sample_df):,} recent posts. {rows_with_views:,} had valid repeat-view measurements.")

    def enrich_item(item, key, occurrence_map, channel_map, domain="", votes=None, is_repo=False):
        context = item.get("sample", item.get("preview", ""))
        item.update(trend_for(key, occurrence_map, channel_map))
        item.update(category_for(domain or key, context, votes))
        item.update(view_metrics(key, is_repo=is_repo))
        item["unique_links"] = len(domain_urls.get(key, set())) if not is_repo else None
        item["unique_channels"] = len(channel_map.get(key, set()))
        return item

    ranked_ai = rank_items(ai_counter, ai_channels)
    top_ai_all = []
    for rank_score, dom, count, unique_ch in ranked_ai:
        meta = ai_meta.get(dom, {})
        item = {
            "title": dom, "url": meta.get("url", f"https://{dom}"),
            "mentions": count, "unique_channels": unique_ch,
            "score": meta.get("score", round(rank_score, 2)),
            "preview": meta.get("sample", ""), "channel": meta.get("channel", ""),
        }
        top_ai_all.append(enrich_item(item, dom, domain_occurrences, domain_channels, dom, domain_category_votes[dom]))

    ranked_web = rank_items(web_counter, web_channels)
    top_web_all = []
    for rank_score, dom, count, unique_ch in ranked_web:
        meta = web_meta.get(dom, {})
        item = {
            "title": dom, "url": meta.get("url", f"https://{dom}"),
            "mentions": count, "unique_channels": unique_ch,
            "score": round(rank_score, 2), "preview": meta.get("sample", ""),
            "channel": meta.get("channel", ""),
        }
        top_web_all.append(enrich_item(item, dom, domain_occurrences, domain_channels, dom, domain_category_votes[dom]))

    ranked_github = rank_items(github_counter, github_channels)
    top_github_all = []
    for rank_score, repo, count, unique_ch in ranked_github:
        meta = github_meta.get(repo, {})
        item = {
            "title": repo, "url": meta.get("url", f"https://github.com/{repo}"),
            "mentions": count, "unique_channels": unique_ch,
            "score": round(rank_score, 2), "preview": meta.get("sample", ""),
            "channel": meta.get("channel", ""),
        }
        top_github_all.append(enrich_item(item, repo, repo_occurrences, github_channels, "github.com", None, True))

    domain_discoveries = {}
    for dom, mentions in domain_counter.items():
        meta = domain_meta.get(dom, {})
        item = {
            "title": dom,
            "domain": dom,
            "url": meta.get("url", f"https://{dom}"),
            "mentions": mentions,
            "unique_channels": len(domain_channels.get(dom, set())),
            "unique_links": len(domain_urls.get(dom, set())),
            "sample": meta.get("sample", ""),
            "preview": meta.get("sample", ""),
            "channel": meta.get("channel", ""),
            "date": meta.get("date", ""),
            "score": round(mentions * 0.6 + len(domain_channels.get(dom, set())) * 0.4, 2),
        }
        item.update(trend_for(dom, domain_occurrences, domain_channels))
        item.update(category_for(dom, item["sample"], domain_category_votes[dom]))
        item.update(view_metrics(dom))
        domain_discoveries[dom] = item

    top_100_domains = [
        domain_discoveries[ranked["domain"]]
        for ranked in rank_top_domains(domain_counter, domain_channels)
        if ranked["domain"] in domain_discoveries
    ]

    candidates = top_ai_all + top_web_all + top_github_all + list(domain_discoveries.values())

    def unique_candidates(items):
        selected = {}
        for candidate in items:
            identity = str(candidate.get("domain") or candidate.get("title") or candidate.get("url") or "").rstrip("/").casefold()
            if not identity:
                identity = str(candidate.get("url", "")).casefold()
            previous = selected.get(identity)
            candidate_key = (
                int(candidate.get("trend_score") or 0),
                int(candidate.get("unique_channels") or 0),
                int(candidate.get("mentions") or 0),
            )
            previous_key = (
                int(previous.get("trend_score") or 0),
                int(previous.get("unique_channels") or 0),
                int(previous.get("mentions") or 0),
            ) if previous else (-1, -1, -1)
            if previous is None or candidate_key > previous_key:
                selected[identity] = candidate
        return list(selected.values())

    deduped_candidates = unique_candidates(candidates)
    top_rising = sorted(
        (
            item.copy() for item in deduped_candidates
            if item.get("trend_status") in {"trending", "rising"}
        ),
        key=lambda item: (
            item.get("trend_score", 0),
            item.get("growth_ratio") or 0,
            item.get("unique_channels", 0),
            item.get("mentions", 0),
        ),
        reverse=True,
    )[:40]
    top_new = sorted(
        (item.copy() for item in deduped_candidates if item.get("trend_status") == "new"),
        key=lambda item: (item.get("latest_post_at", ""), item.get("mentions", 0)),
        reverse=True,
    )[:40]

    explorer_payload = {
        "updated_at": now_utc.isoformat(),
        "scan_window_posts": int(len(sample_df)),
        "trend_settings": {
            "primary_window_hours": 6,
            "baseline_window_hours": 24,
            "minimum_trending_mentions": 5,
            "minimum_trending_channels": 3,
            "note": "Trend is based on Telegram post publication time. View velocity is reported separately and only after repeated snapshots."
        },
        "top_ai_tools": top_ai_all[:35],
        "top_web_gems": top_web_all[:35],
        "top_repos": top_github_all[:35],
        "top_packages": [{"package": package, "mentions": count} for package, count in tools_counter.most_common(40)],
        "top_100_domains": top_100_domains,
        "top_rising": top_rising,
        "top_new": top_new,
        "view_tracking": {
            "posts_with_valid_repeat_snapshots": rows_with_views,
            "source": "Telegram source-post view counts, not destination website analytics",
        },
    }

    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/curated_explorer_data"
    headers = {"Authorization": f"Bearer {cf_token}", "Content-Type": "application/json"}
    res = requests.put(url, headers=headers, json=explorer_payload, timeout=30)
    res.raise_for_status()
    result = res.json()
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError(f"Cloudflare rejected Explorer update: {result.get('errors', result)}")
    print(
        f"[✓] Explorer Smart-Pushed: {len(top_ai_all[:35])} AI Tools, "
        f"{len(top_web_all[:35])} Web Gems, {len(top_github_all[:35])} Repos, "
        f"{len(top_rising)} rising and {len(top_new)} new discoveries."
    )

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

    cols_to_keep = [
        "id", "username", "post_id", "text", "image", "video", "media_type", "views", "date",
        "scraped_at", "previous_views", "previous_scraped_at", "views_delta",
        "views_velocity_per_hour", "views_interval_hours",
    ]
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



def attach_view_history(existing_df, new_df):
    """Attach repeat-scrape view deltas to newly fetched posts without faking history."""
    if new_df.empty:
        return new_df
    enriched = new_df.copy()
    now_text = datetime.now(timezone.utc).isoformat()
    if "scraped_at" not in enriched.columns:
        enriched["scraped_at"] = now_text
    else:
        empty_stamp = enriched["scraped_at"].isna() | enriched["scraped_at"].astype(str).str.strip().eq("")
        enriched.loc[empty_stamp, "scraped_at"] = now_text

    existing_by_id = {}
    if not existing_df.empty and "id" in existing_df.columns:
        for _, old_row in existing_df.iterrows():
            old_id = str(old_row.get("id", "") or "")
            if old_id:
                existing_by_id[old_id] = {
                    "views": old_row.get("views"),
                    "scraped_at": old_row.get("scraped_at"),
                }

    previous_views = []
    previous_scraped_at = []
    views_delta = []
    velocity_per_hour = []
    interval_hours_values = []
    for _, row in enriched.iterrows():
        old = existing_by_id.get(str(row.get("id", "") or ""))
        old_views = None
        old_stamp = ""
        current_views = None
        try:
            if old and pd.notna(old.get("views")):
                parsed_old_views = float(old["views"])
                old_views = parsed_old_views if math.isfinite(parsed_old_views) else None
            if pd.notna(row.get("views")):
                parsed_current_views = float(row["views"])
                current_views = parsed_current_views if math.isfinite(parsed_current_views) else None
        except (TypeError, ValueError):
            old_views, current_views = None, None

        old_stamp_value = old.get("scraped_at") if old else None
        parsed_old = parse_utc_datetime(old_stamp_value)
        parsed_current = parse_utc_datetime(row.get("scraped_at"))
        old_stamp = parsed_old.isoformat() if parsed_old else ""
        delta_value = rate_value = interval_value = None
        if old_views is not None and current_views is not None and parsed_old and parsed_current:
            interval = (parsed_current - parsed_old).total_seconds() / 3600
            delta = current_views - old_views
            # Ignore very short, very stale, or negative-rounded measurements.
            if 0.1 <= interval <= 24 and delta >= 0:
                delta_value = int(delta)
                rate_value = round(delta / interval, 2)
                interval_value = round(interval, 2)

        previous_views.append(int(old_views) if old_views is not None else None)
        previous_scraped_at.append(old_stamp)
        views_delta.append(delta_value)
        velocity_per_hour.append(rate_value)
        interval_hours_values.append(interval_value)

    enriched["previous_views"] = previous_views
    enriched["previous_scraped_at"] = previous_scraped_at
    enriched["views_delta"] = views_delta
    enriched["views_velocity_per_hour"] = velocity_per_hour
    enriched["views_interval_hours"] = interval_hours_values
    return enriched


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
            chunk_new = attach_view_history(existing_year_df, chunk_new)

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
