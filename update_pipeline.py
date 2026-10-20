import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

HF_REPO_ID = os.getenv("HF_REPO_ID", "Moo408/dat-9f82c4")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_DIR = Path("data_sync")
CRAWLER_DIR = Path("crawler_output")
KEYWORDS_FILE = Path("interests.json")

# The HTML frontend depends on these exact columns and their names.
FEED_COLUMNS = [
    "id", "username", "post_id", "text", "image",
    "video", "media_type", "views", "date",
]

# Feed selection limits. The final feed is selected from all available year
# partitions, not only the 2026 partition.
MAX_CANDIDATES_PER_PARTITION = 15000
MAX_FEED_POSTS = 3000

# Keep the serialized KV value below the documented KV value-size limit.
# Leave a margin for JSON escaping and future small changes.
MAX_KV_VALUE_BYTES = 22 * 1024 * 1024

# Reject posts dominated by Chinese characters, instead of rejecting a post
# because it contains even one Chinese character.
CHINESE_LETTER_RATIO_LIMIT = 0.35

# Score threshold for the feed. A matching keyword is still required.
MIN_QUALITY_SCORE = 2

# Low-value promotional signals reduce the score. These are not hard blocks.
SOFT_SPAM_TERMS = [
    "bonus upto", "free paytm", "earn money fast", "work from home",
    "limited offer", "join my channel", "subscribe now", "dm for promotion",
    "airdrop", "giveaway", "referral bonus",
]

# Definite spam/gambling indicators. Word boundaries prevent matches such as
# "casino" inside a longer unrelated word. URL patterns remain literal.
HARD_SPAM_REGEXES = [
    r"(?<!\w)₹",
    r"(?<!\w)yono(?!\w)",
    r"(?<!\w)rummy(?!\w)",
    r"(?<!\w)bhailog(?!\w)",
    r"(?<!\w)bhai(?!\w)",
    r"(?<!\w)loot(?!\w)",
    r"(?<!\w)1xbet(?!\w)",
    r"(?<!\w)betwinner(?!\w)",
    r"(?<!\w)melbet(?!\w)",
    r"(?<!\w)casino(?!\w)",
    r"(?<!\w)poker(?!\w)",
    r"t\.me/\+",
    r"t\.me/joinchat",
    r"addlist/",
]

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
URL_RE = re.compile(r"https?://\S+|t\.me/\S+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}", re.IGNORECASE)

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
api = HfApi(token=HF_TOKEN)


# =============================================================================
# 2. SHARED HELPERS
# =============================================================================

def log(message):
    """Print timestamped messages to make workflow logs easier to inspect."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"[{now}] {message}", flush=True)


def normalize_text(value):
    """Normalize Unicode, whitespace and case for matching only."""
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = SPACE_RE.sub(" ", value).strip().casefold()
    return value


def normalize_username(value):
    """Normalize a channel username or a Telegram channel URL."""
    value = normalize_text(value)
    value = value.removeprefix("https://").removeprefix("http://")
    value = value.removeprefix("www.")
    value = value.removeprefix("t.me/")
    value = value.strip().strip("/")
    value = value.removeprefix("@")
    return value


def safe_text(value):
    """Convert pandas missing values into an empty string."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_dates(series):
    """Parse dates consistently; invalid dates become NaT and sort last."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def read_remote_parquet(filename):
    """
    Download and read one remote Parquet file.
    Return an empty DataFrame only when the remote file is genuinely absent.
    Network/authentication errors are raised to protect existing data.
    """
    try:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except EntryNotFoundError:
        log(f"Remote file does not exist yet: {filename}")
        return pd.DataFrame()

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read remote Parquet file {filename}: {exc}") from exc


def upload_file(local_path, remote_path, message):
    """Upload a file to the existing Hugging Face dataset repository."""
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=message,
    )


def compile_keyword_patterns(keywords):
    """
    Compile each keyword separately.
    This avoids one enormous regex and supports phrases, punctuation and
    keyword matching across English/Arabic text.
    """
    patterns = []
    for keyword in keywords:
        keyword = normalize_text(keyword)
        if not keyword:
            continue

        escaped = re.escape(keyword)
        starts_with_word = keyword[0].isalnum() or "\u0600" <= keyword[0] <= "\u06ff"
        ends_with_word = keyword[-1].isalnum() or "\u0600" <= keyword[-1] <= "\u06ff"

        prefix = r"(?<![\w])" if starts_with_word else ""
        suffix = r"(?![\w])" if ends_with_word else ""
        try:
            patterns.append((keyword, re.compile(prefix + escaped + suffix, re.IGNORECASE)))
        except re.error as exc:
            log(f"Skipping invalid keyword {keyword!r}: {exc}")
    return patterns


# =============================================================================
# 3. CENTRAL BLACKLIST
# =============================================================================

def get_cloudflare_blacklist():
    """Fetch the channel denylist stored under the KV key 'blacklist'."""
    cf_token = os.getenv("CF_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_kv_id = os.getenv("CF_KV_NAMESPACE_ID")

    if not (cf_token and cf_account and cf_kv_id):
        log("Cloudflare blacklist unavailable: one or more secrets are missing.")
        return set()

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/blacklist"
    )
    headers = {"Authorization": f"Bearer {cf_token}"}

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 404:
            log("Cloudflare blacklist key not found. Continuing with an empty list.")
            return set()
        response.raise_for_status()

        # KV returns the stored value as the response body, not a standard
        # Cloudflare {success, result} envelope.
        payload = response.json()
        if not isinstance(payload, list):
            log("Blacklist value is not a JSON array. Ignoring it safely.")
            return set()

        result = {normalize_username(item) for item in payload if normalize_username(item)}
        log(f"Loaded {len(result):,} blocked channel usernames.")
        return result
    except Exception as exc:
        # Fail closed for purging: stop instead of silently skipping a blacklist
        # and accidentally re-uploading content that should have been removed.
        raise RuntimeError(f"Could not read Cloudflare blacklist: {exc}") from exc


# =============================================================================
# 4. KEYWORDS AND SMART POST FILTER
# =============================================================================

def load_user_keywords():
    """Load the exact user-maintained keyword list from interests.json."""
    if not KEYWORDS_FILE.exists():
        log("interests.json is missing. No posts will be selected for the feed.")
        return []

    try:
        with KEYWORDS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        raw = data.get("keywords", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            log('interests.json must contain a list under the "keywords" key.')
            return []

        seen = set()
        keywords = []
        for item in raw:
            keyword = normalize_text(item)
            if keyword and keyword not in seen:
                seen.add(keyword)
                keywords.append(keyword)

        log(f"Loaded {len(keywords):,} unique keywords from interests.json.")
        return keywords
    except Exception as exc:
        raise RuntimeError(f"Could not parse interests.json: {exc}") from exc


def chinese_ratio(text):
    """Return the fraction of characters that are Chinese among all letters."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if CHINESE_RE.fullmatch(char)) / len(letters)


def calculate_quality_score(text, username, matched_keywords):
    """
    Transparent rule-based scoring. A keyword match is mandatory.
    The score is internal and is NOT added to the frontend JSON schema.
    """
    score = min(len(matched_keywords), 5) * 2
    text_lower = normalize_text(text)
    username_lower = normalize_username(username)

    # Helpful technical signals.
    technical_signals = [
        "github.com/", "gitlab.com/", "pypi.org/", "npmjs.com/",
        "pip install", "npm install", "docker", "api", "sdk",
        "tutorial", "how to", "example", "documentation", "docs",
        "open source", "repository", "repo", "release", "benchmark",
        "python", "javascript", "typescript", "rust", "sql",
    ]
    for signal in technical_signals:
        if signal in text_lower:
            score += 2
            break

    # A URL by itself is not spam. A repository/documentation link is useful.
    if re.search(r"github\.com/|gitlab\.com/|docs\.", text_lower):
        score += 2

    # Soft promotional phrases reduce score, but do not auto-delete the post.
    for term in SOFT_SPAM_TERMS:
        if term in text_lower:
            score -= 3

    # Repeated characters are a weak spam signal, so apply a small penalty.
    if REPEATED_CHAR_RE.search(text):
        score -= 1

    return score


def curate_posts(posts_df):
    """
    Return a clean, ranked feed with exactly FEED_COLUMNS.
    Hard blocks are applied first. Remaining posts need a keyword match and
    the configured minimum score. Quality fields stay internal.
    """
    if posts_df.empty:
        return []

    keywords = load_user_keywords()
    if not keywords:
        return []

    keyword_patterns = compile_keyword_patterns(keywords)
    if not keyword_patterns:
        return []

    hard_spam_pattern = re.compile(
        "|".join(f"(?:{pattern})" for pattern in HARD_SPAM_REGEXES),
        re.IGNORECASE,
    )

    frame = posts_df.copy()
    if "date" not in frame.columns:
        frame["date"] = pd.NaT
    frame["_parsed_date"] = parse_dates(frame["date"])

    # Latest records first; invalid dates are placed at the end.
    frame = frame.sort_values("_parsed_date", ascending=False, na_position="last")
    dedupe_columns = [column for column in ["username", "post_id"] if column in frame.columns]
    if dedupe_columns:
        frame = frame.drop_duplicates(subset=dedupe_columns, keep="first")

    selected = []
    counters = {
        "empty": 0,
        "spam": 0,
        "chinese": 0,
        "no_keyword": 0,
        "low_score": 0,
    }

    for record in frame.to_dict(orient="records"):
        post_text = safe_text(record.get("text"))
        username = safe_text(record.get("username"))

        if not post_text or post_text.casefold() == "nan":
            counters["empty"] += 1
            continue

        combined_for_spam = f"{post_text} {username}"
        if hard_spam_pattern.search(combined_for_spam):
            counters["spam"] += 1
            continue

        if chinese_ratio(post_text) > CHINESE_LETTER_RATIO_LIMIT:
            counters["chinese"] += 1
            continue

        normalized = normalize_text(post_text)
        matches = [keyword for keyword, pattern in keyword_patterns if pattern.search(normalized)]
        if not matches:
            counters["no_keyword"] += 1
            continue

        score = calculate_quality_score(post_text, username, matches)
        if score < MIN_QUALITY_SCORE:
            counters["low_score"] += 1
            continue

        record["_quality_score"] = score
        record["_matched_keywords"] = matches
        record["_parsed_date"] = record.get("_parsed_date")
        selected.append(record)

    # Rank by quality first, then by recency.
    selected.sort(
        key=lambda row: (
            row.get("_quality_score", 0),
            row.get("_parsed_date") if pd.notna(row.get("_parsed_date")) else pd.Timestamp.min.tz_localize("UTC"),
        ),
        reverse=True,
    )

    log(
        "Filter results: "
        f"accepted={len(selected):,}, empty={counters['empty']:,}, "
        f"spam={counters['spam']:,}, Chinese-dominated={counters['chinese']:,}, "
        f"no-keyword={counters['no_keyword']:,}, low-score={counters['low_score']:,}"
    )
    return selected


# =============================================================================
# 5. CLOUDflare FEED EXPORT
# =============================================================================

def push_to_cloudflare_kv(posts_df):
    """
    Write selected posts to the same KV key and preserve the exact frontend
    schema: id, username, post_id, text, image, video, media_type, views, date.
    """
    cf_token = os.getenv("CF_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_kv_id = os.getenv("CF_KV_NAMESPACE_ID")

    if not (cf_token and cf_account and cf_kv_id):
        log("Skipping feed export: Cloudflare secrets are missing.")
        return

    if posts_df.empty:
        log("No candidate posts available for feed export.")
        return

    # Bound candidate processing per partition before running text matching.
    candidates = posts_df.copy()
    if "date" in candidates.columns:
        candidates["_sort_date"] = parse_dates(candidates["date"])
        candidates = candidates.sort_values("_sort_date", ascending=False, na_position="last")
        candidates = candidates.drop(columns=["_sort_date"])
    candidates = candidates.head(MAX_CANDIDATES_PER_PARTITION)

    curated = curate_posts(candidates)
    curated = curated[:MAX_FEED_POSTS]

    # Preserve the original HTML schema. Internal scoring fields are omitted.
    feed = []
    for post in curated:
        item = {}
        for column in FEED_COLUMNS:
            value = post.get(column, "")
            if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
                value = ""
            # Keep the original behavior of serializing values as strings.
            item[column] = str(value)
        feed.append(item)

    # Cloudflare KV has a per-value size limit. Reduce the number of posts until
    # the JSON value fits under a conservative 22 MiB cap.
    while feed:
        payload = json.dumps(feed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(payload) <= MAX_KV_VALUE_BYTES:
            break
        feed.pop()
    if not feed:
        log("No posts fit within the configured Cloudflare KV value-size limit.")
        return

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{cf_account}/storage/kv/namespaces/{cf_kv_id}/values/latest_posts"
    )
    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    try:
        response = requests.put(url, headers=headers, data=payload, timeout=60)
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Cloudflare KV upload failed ({response.status_code}): {response.text}")
        log(f"Uploaded {len(feed):,} posts to latest_posts ({len(payload) / 1024:.1f} KiB).")
    except Exception as exc:
        raise RuntimeError(f"Could not upload latest_posts to Cloudflare KV: {exc}") from exc


# =============================================================================
# 6. CHANNEL DATABASE SYNC
# =============================================================================

def sync_channels(blacklist):
    """Merge crawler_output/c.parquet into the existing c.parquet archive."""
    new_path = CRAWLER_DIR / "c.parquet"
    new_df = pd.read_parquet(new_path) if new_path.exists() else pd.DataFrame()
    existing_df = read_remote_parquet("c.parquet")

    merged = pd.concat([existing_df, new_df], ignore_index=True)
    if merged.empty:
        log("No channel rows to sync.")
        return len(new_df), 0, 0

    if "username" not in merged.columns:
        raise RuntimeError("c.parquet has no 'username' column; refusing to upload.")

    merged["_normalized_username"] = merged["username"].map(normalize_username)
    merged = merged.drop_duplicates(subset=["_normalized_username"], keep="last")

    removed = 0
    if blacklist:
        before = len(merged)
        merged = merged[~merged["_normalized_username"].isin(blacklist)]
        removed = before - len(merged)
    merged = merged.drop(columns=["_normalized_username"])

    output = LOCAL_DIR / "c.parquet"
    merged.to_parquet(output, compression="zstd", index=False)
    upload_file(output, "c.parquet", f"Sync channels: {len(merged):,} rows, purged {removed:,}")

    checked = 0
    if "last_checked" in merged.columns:
        checked = merged["last_checked"].notna() & merged["last_checked"].astype(str).str.strip().ne("")
        checked = int(checked.sum())

    remaining = len(merged) - checked
    log(f"Channels synced: new={len(new_df):,}, total={len(merged):,}, purged={removed:,}.")
    return len(new_df), len(merged), remaining


# =============================================================================
# 7. YEAR-PARTITIONED POST DATABASE SYNC
# =============================================================================

def extract_year_safely(value):
    """Extract a plausible year from a date value."""
    if value is None or pd.isna(value) or not str(value).strip():
        return "unknown"
    parsed = pd.to_datetime(str(value).strip(), errors="coerce", utc=True)
    if pd.notna(parsed) and 2000 <= parsed.year <= 2100:
        return str(parsed.year)
    match = re.match(r"^\s*(20\d{2})", str(value))
    if match:
        return match.group(1)
    return "unknown"


def load_manifest():
    """Load manifest.json, creating a starter manifest only if the file is absent."""
    try:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="manifest.json",
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except EntryNotFoundError:
        log("manifest.json is missing. Starting a new manifest.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("manifest root must be a JSON object")
        return value
    except Exception as exc:
        raise RuntimeError(f"Could not read manifest.json: {exc}") from exc


def sync_partitioned_posts(blacklist):
    """Merge new posts into year partitions and refresh the manifest."""
    new_path = CRAWLER_DIR / "p.parquet"
    new_posts = pd.read_parquet(new_path) if new_path.exists() else pd.DataFrame()

    if not new_posts.empty and "date" not in new_posts.columns:
        raise RuntimeError("crawler_output/p.parquet has no 'date' column.")

    if not new_posts.empty:
        new_posts["_year_partition"] = new_posts["date"].map(extract_year_safely)

    manifest = load_manifest()

    # Ensure each year present in the incoming crawler data has a partition.
    if not new_posts.empty:
        for year in new_posts["_year_partition"].dropna().unique():
            manifest.setdefault(str(year), {"file": f"posts_{year}.parquet"})

    # Keep a bounded sample from each partition for the latest-post feed.
    feed_samples = []
    total_posts = 0

    for year, meta in sorted(manifest.items(), key=lambda pair: str(pair[0])):
        partition_filename = meta.get("file", f"posts_{year}.parquet")
        existing_year = read_remote_parquet(partition_filename)

        if not new_posts.empty:
            chunk = new_posts[new_posts["_year_partition"].astype(str) == str(year)]
            chunk = chunk.drop(columns=["_year_partition"], errors="ignore")
        else:
            chunk = pd.DataFrame()

        merged = pd.concat([existing_year, chunk], ignore_index=True)
        if merged.empty:
            manifest[year] = {
                "file": partition_filename,
                "total_rows": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            continue

        # Prefer id as the unique key. Fall back to username + post_id.
        if "id" in merged.columns:
            ids = merged["id"].astype(str).str.strip()
            has_id = ids.ne("") & ids.ne("nan")
            with_id = merged[has_id].drop_duplicates(subset=["id"], keep="last")
            without_id = merged[~has_id]
            if {"username", "post_id"}.issubset(without_id.columns):
                without_id = without_id.drop_duplicates(subset=["username", "post_id"], keep="last")
            merged = pd.concat([with_id, without_id], ignore_index=True)
        elif {"username", "post_id"}.issubset(merged.columns):
            merged = merged.drop_duplicates(subset=["username", "post_id"], keep="last")

        purged = 0
        if blacklist and "username" in merged.columns:
            before = len(merged)
            normalized_users = merged["username"].map(normalize_username)
            merged = merged[~normalized_users.isin(blacklist)]
            purged = before - len(merged)

        if purged:
            log(f"Purged {purged:,} blocked posts from {partition_filename}.")

        # Keep a date-sorted candidate sample from this partition for feed creation.
        sample = merged.copy()
        if "date" in sample.columns:
            sample["_sort_date"] = parse_dates(sample["date"])
            sample = sample.sort_values("_sort_date", ascending=False, na_position="last")
            sample = sample.drop(columns=["_sort_date"])
        feed_samples.append(sample.head(MAX_CANDIDATES_PER_PARTITION))

        local_file = LOCAL_DIR / partition_filename
        merged.to_parquet(local_file, compression="zstd", index=False)
        upload_file(
            local_file,
            partition_filename,
            f"Sync and clean {partition_filename}: {len(merged):,} rows",
        )

        manifest[year] = {
            "file": partition_filename,
            "total_rows": int(len(merged)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        total_posts += len(merged)
        log(f"Partition synced: {partition_filename}, rows={len(merged):,}.")

    manifest_path = LOCAL_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    upload_file(manifest_path, "manifest.json", "Update dataset partitions manifest")

    if feed_samples:
        feed_candidates = pd.concat(feed_samples, ignore_index=True)
        if "date" in feed_candidates.columns:
            feed_candidates["_sort_date"] = parse_dates(feed_candidates["date"])
            feed_candidates = feed_candidates.sort_values(
                "_sort_date", ascending=False, na_position="last"
            )
            feed_candidates = feed_candidates.drop(columns=["_sort_date"])
        # Global cap after combining all year samples.
        feed_candidates = feed_candidates.head(MAX_CANDIDATES_PER_PARTITION)
        push_to_cloudflare_kv(feed_candidates)

    return len(new_posts), total_posts


# =============================================================================
# 8. MAIN ENTRY POINT
# =============================================================================

def main():
    log("=" * 68)
    log("Starting dataset sync, blacklist purge and smart post curation.")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing. Set it as a repository secret.")

    blacklist = get_cloudflare_blacklist()
    channels_new, channels_total, channels_remaining = sync_channels(blacklist)
    posts_new, posts_total = sync_partitioned_posts(blacklist)

    checked = channels_total - channels_remaining
    progress = (checked / channels_total * 100) if channels_total else 0.0

    print("\n" + "=" * 68)
    print("SYNC REPORT")
    print("=" * 68)
    print(f"New/updated channel rows : {channels_new:,}")
    print(f"New crawler post rows    : {posts_new:,}")
    print(f"Total channels           : {channels_total:,}")
    print(f"Checked channels         : {checked:,}")
    print(f"Channels remaining       : {channels_remaining:,}")
    print(f"Queue completion         : {progress:.2f}%")
    print(f"Total archived posts     : {posts_total:,}")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"FATAL: {error}")
        sys.exit(1)
