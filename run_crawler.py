import os
import re
import asyncio
import aiohttp
import random
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timezone
from huggingface_hub import hf_hub_download

# ==================== الإعدادات ====================
CHANNELS_TO_SCRAPE_PER_RUN = 1500  # فحص 1,500 قناة في كل جولة
CONCURRENT_REQUESTS = 12          # سرعة توازي متزنة وآمنة
OUTPUT_DIR = "crawler_output"
# ===================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

def extract_channel_meta(soup, username):
    title = username
    description = ""
    subscribers = "0"
    avatar_url = ""
    is_verified = False

    try:
        title_el = soup.select_one(".tgme_channel_info_header_title span")
        if title_el:
            title = title_el.get_text(strip=True)

        desc_el = soup.select_one(".tgme_channel_info_description")
        if desc_el:
            description = desc_el.get_text(separator="\n", strip=True)

        counters = soup.select(".tgme_channel_info_counter .counter_value")
        for c in counters:
            val = c.get_text(strip=True)
            parent_text = c.parent.get_text().lower()
            if "subscribers" in parent_text or "members" in parent_text:
                if "K" in val:
                    subscribers = str(int(float(val.replace("K", "")) * 1000))
                elif "M" in val:
                    subscribers = str(int(float(val.replace("M", "")) * 1000000))
                else:
                    subscribers = re.sub(r"[^\d]", "", val) or "0"

        avatar_el = soup.select_one(".tgme_page_photo_image img")
        if avatar_el and avatar_el.get("src"):
            avatar_url = avatar_el["src"]

        if soup.select_one(".verified-icon"):
            is_verified = True
    except Exception:
        pass

    return {
        "username": str(username).lower(),
        "title": str(title),
        "description": str(description),
        "subscribers": str(subscribers),
        "avatar": str(avatar_url),
        "verified": str(is_verified),
        "last_checked": datetime.now(timezone.utc).isoformat()
    }

def extract_post_data(msg, username):
    try:
        post_div = msg.select_one(".tgme_widget_message")
        if not post_div:
            return None, set()

        data_post = post_div.get("data-post", "")
        if not data_post or "/" not in data_post:
            return None, set()

        post_id = data_post.split("/")[-1]
        unique_id = f"{username}_{post_id}"

        # استخراج النص مع الحفاظ على الروابط التشعبية المخفية في الكلمات
        text_el = msg.select_one(".tgme_widget_message_text")
        text = ""
        discovered = set()

        if text_el:
            for a_tag in text_el.select("a"):
                href = a_tag.get("href", "")
                link_text = a_tag.get_text(strip=True)
                # إذا كانت الكلمة تحوي رابطاً داخلياً ومختلفاً عن النص المعروض (مثل: حمّل الآن)
                if href and href != link_text and not link_text.startswith("@") and not link_text.startswith("#"):
                    a_tag.replace_with(f"{link_text} ({href})")
            
            text = text_el.get_text(separator="\n", strip=True)

        if text:
            mentions = re.findall(r"@([a-zA-Z0-9_]{4,32})", text)
            for m in mentions:
                discovered.add(m.lower())

        images = []
        video_url = ""
        media_type = "text"

        photo_thumbs = msg.select(".tgme_widget_message_photo_wrap")
        for thumb in photo_thumbs:
            style = thumb.get("style", "")
            match = re.search(r"background-image:url\('([^']+)'\)", style)
            if match:
                images.append(match.group(1))

        video_el = msg.select_one("video")
        if video_el and video_el.get("src"):
            video_url = video_el["src"]
            media_type = "video"
        elif images:
            media_type = "image"

        views = "0"
        views_el = msg.select_one(".tgme_widget_message_views")
        if views_el:
            v_text = views_el.get_text(strip=True)
            if "K" in v_text:
                views = str(int(float(v_text.replace("K", "")) * 1000))
            elif "M" in v_text:
                views = str(int(float(v_text.replace("M", "")) * 1000000))
            else:
                views = re.sub(r"[^\d]", "", v_text) or "0"

        date_str = ""
        time_el = msg.select_one("time")
        if time_el and time_el.get("datetime"):
            date_str = time_el.get("datetime")

        reactions_count = 0
        reaction_els = msg.select(".tgme_widget_message_reactions .tgme_reaction")
        for r in reaction_els:
            count_span = r.select_one(".tgme_reaction_count")
            if count_span:
                val = count_span.get_text(strip=True)
                reactions_count += int(re.sub(r"[^\d]", "", val) or 0)

        return {
            "id": unique_id,
            "username": str(username).lower(),
            "post_id": str(post_id),
            "text": str(text),
            "image": str(images[0] if images else ""),
            "all_images": ",".join(images) if len(images) > 1 else str(images[0] if images else ""),
            "video": str(video_url),
            "media_type": str(media_type),
            "views": str(views),
            "reactions": str(reactions_count),
            "date": str(date_str),
            "scraped_at": datetime.now(timezone.utc).isoformat()
        }, discovered

    except Exception:
        return None, set()

async def fetch_channel(session, semaphore, username):
    url = f"https://t.me/s/{username}"
    async with semaphore:
        for attempt in range(3):
            try:
                req_headers = {"User-Agent": random.choice(USER_AGENTS)}
                async with session.get(url, headers=req_headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        soup = BeautifulSoup(html, "html.parser")
                        channel_meta = extract_channel_meta(soup, username)
                        channel_meta["last_checked"] = datetime.now(timezone.utc).isoformat()
                        posts, discovered_channels = [], set()
                        for msg in soup.select(".tgme_widget_message_wrap"):
                            post_data, found_ch = extract_post_data(msg, username)
                            if post_data:
                                posts.append(post_data)
                            discovered_channels.update(found_ch)
                        return channel_meta, posts, discovered_channels
                    if resp.status not in (429, 500, 502, 503, 504):
                        print(f"[!] {username}: HTTP {resp.status}; leaving channel queued")
                        return None, [], set()
                    retry_after = resp.headers.get("Retry-After", "")
                    delay = min(float(retry_after), 30) if retry_after.isdigit() else 2 ** attempt
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                delay = 2 ** attempt
                if attempt == 2:
                    print(f"[!] {username}: request failed ({exc}); leaving channel queued")
            if attempt < 2:
                await asyncio.sleep(delay + random.uniform(0, 0.5))
    return None, [], set()

async def main():
    print("[*] Starting Tech-Prioritized Queue Telegram Cloud Scraper...")

    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_REPO_ID = os.getenv("HF_REPO_ID", "Moo408/dat-9f82c4")
    target_usernames = []

    try:
        print("[*] Downloading channels queue from Hugging Face...")
        c_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="c.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        c_df = pd.read_parquet(c_path)
        
        # New channels are crawled first. Checked channels are revisited once
        # their snapshot is old enough to measure view changes without re-fetching
        # the same recent channels on every 30-minute run.
        tech_pattern = r'python|termux|github|mod|apk|android|script|linux|code|tool|hack|api|dev'
        now_utc = pd.Timestamp.now(tz="UTC")
        if "last_checked" in c_df.columns:
            checked_at = pd.to_datetime(c_df["last_checked"], utc=True, errors="coerce")
            uninspected_mask = checked_at.isna()
            uninspected = c_df.loc[uninspected_mask].copy()
            checked = c_df.loc[~uninspected_mask].copy()
            checked["_last_checked_ts"] = checked_at.loc[~uninspected_mask]
            refresh_cutoff = now_utc - pd.Timedelta(hours=6)
            stale_checked = checked.loc[checked["_last_checked_ts"] <= refresh_cutoff].copy()
            stale_checked = stale_checked.sort_values("_last_checked_ts", ascending=True, na_position="first")
        else:
            uninspected = c_df.copy()
            stale_checked = c_df.iloc[0:0].copy()

        remaining_count = len(uninspected)
        refresh_count = len(stale_checked)
        print(f"[*] New channels waiting in queue: {remaining_count:,}")
        print(f"[*] Previously inspected channels due for refresh: {refresh_count:,}")

        def prioritize_tech(frame):
            if frame.empty:
                return frame
            frame = frame.copy()
            frame["_is_tech"] = frame["username"].astype(str).str.contains(tech_pattern, case=False, na=False)
            sort_columns = ["_is_tech"]
            ascending = [False]
            if "_last_checked_ts" in frame.columns:
                sort_columns.append("_last_checked_ts")
                ascending.append(True)
            frame = frame.sort_values(sort_columns, ascending=ascending, na_position="first")
            return frame.drop(columns=["_is_tech", "_last_checked_ts"], errors="ignore")

        # Uninspected channels have priority, followed by the oldest six-hour
        # snapshots. Avoid silently stopping once the initial queue is exhausted.
        priority_queue = pd.concat(
            [prioritize_tech(uninspected), prioritize_tech(stale_checked)],
            ignore_index=True,
        )
        if "username" in priority_queue.columns:
            priority_queue = priority_queue.drop_duplicates(subset=["username"], keep="first")
            target_usernames = priority_queue["username"].dropna().astype(str).head(CHANNELS_TO_SCRAPE_PER_RUN).tolist()
        print(f"[+] Loaded {len(target_usernames)} target channels for this run.");
    except Exception as e:
        print(f"[-] Could not load channels queue: {e}")

    if not target_usernames:
        print("[-] No channels to process this run.")
        return

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    all_new_posts = []
    discovered_all = set()
    channels_meta = []

    conn = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [fetch_channel(session, semaphore, u) for u in target_usernames]
        results = await asyncio.gather(*tasks)

        for meta, posts, discovered in results:
            if meta:
                channels_meta.append(meta)
            if posts:
                all_new_posts.extend(posts)
            discovered_all.update(discovered)

    print(f"[+] Total posts scraped: {len(all_new_posts)}")
    print(f"[+] Inspected channels recorded: {len(channels_meta)}")
    print(f"[+] Discovered new channels: {len(discovered_all)}")

    if all_new_posts:
        posts_df = pd.DataFrame(all_new_posts).drop_duplicates(subset=["id"], keep="last")
        posts_out = os.path.join(OUTPUT_DIR, "p.parquet")
        posts_df.to_parquet(posts_out, compression="zstd", index=False)
        print(f"[+] Saved scraped posts to {posts_out}")

    new_channels_records = [{"username": str(d), "title": str(d), "description": "", "subscribers": "0", "avatar": "", "verified": "False", "last_checked": ""} for d in discovered_all]
    all_channels = channels_meta + new_channels_records
    if all_channels:
        ch_df = pd.DataFrame(all_channels).drop_duplicates(subset=["username"], keep="first")
        ch_out = os.path.join(OUTPUT_DIR, "c.parquet")
        ch_df.to_parquet(ch_out, compression="zstd", index=False)
        print(f"[+] Saved channel updates to {ch_out}")

if __name__ == "__main__":
    asyncio.run(main())
