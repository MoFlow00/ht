import os
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from huggingface_hub import hf_hub_download

# ==================== إعدادات الكراولر ====================
CHANNELS_TO_SCRAPE_PER_RUN = 500   # عدد القنوات التي يتم فحصها بكل دورة
CONCURRENT_REQUESTS = 10           # سرعة التوازي
OUTPUT_DIR = "crawler_output"
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def extract_post_data(soup, username):
    posts_data = []
    discovered_channels = set()
    
    channel_title = ""
    title_tag = soup.select_one(".tgme_channel_info_header_title span")
    if title_tag:
        channel_title = title_tag.get_text(strip=True)

    messages = soup.select(".tgme_widget_message_wrap")
    for msg in messages:
        try:
            post_div = msg.select_one(".tgme_widget_message")
            if not post_div:
                continue
            data_post = post_div.get("data-post", "")
            if not data_post or "/" not in data_post:
                continue
            post_id = data_post.split("/")[-1]

            text_el = msg.select_one(".tgme_widget_message_text")
            text = text_el.get_text(separator="\n", strip=True) if text_el else ""

            if text:
                mentions = re.findall(r"@([a-zA-Z0-9_]{4,32})", text)
                for m in mentions:
                    discovered_channels.add(m.lower())

            image_url = ""
            video_url = ""
            media_type = "text"

            photo_thumb = msg.select_one(".tgme_widget_message_photo_wrap")
            if photo_thumb:
                style = photo_thumb.get("style", "")
                match = re.search(r"background-image:url\('([^']+)'\)", style)
                if match:
                    image_url = match.group(1)
                    media_type = "image"

            video_el = msg.select_one("video")
            if video_el:
                video_url = video_el.get("src", "")
                media_type = "video"

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

            posts_data.append({
                "username": str(username),
                "post_id": str(post_id),
                "text": str(text),
                "image": str(image_url),
                "video": str(video_url),
                "media_type": str(media_type),
                "views": str(views),
                "date": str(date_str)
            })
        except Exception:
            continue

    return channel_title, posts_data, discovered_channels

async def fetch_channel(session, semaphore, username):
    url = f"https://t.me/s/{username}"
    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=12) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    return extract_post_data(soup, username)
        except Exception:
            pass
    return None, [], set()

async def main():
    print("[*] Telegram Cloud Scraper started...")

    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_REPO_ID = os.getenv("HF_REPO_ID")
    target_usernames = []

    # تنزيل ملف القنوات من السحابة لسحب 500 قناة عشوائية حقيقية
    try:
        print("[*] Fetching channel list from Hugging Face...")
        c_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="c.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        c_df = pd.read_parquet(c_path, columns=["username"])
        if "username" in c_df.columns:
            target_usernames = c_df["username"].dropna().sample(min(CHANNELS_TO_SCRAPE_PER_RUN, len(c_df))).tolist()
            print(f"[+] Successfully loaded {len(target_usernames)} target channels from HF!")
    except Exception as e:
        print(f"[-] Could not load channels from HF: {e}")

    if not target_usernames:
        target_usernames = ["telegram", "durov", "tginfo", "techcrunch", "pythonres"]

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    all_new_posts = []
    discovered_all = set()
    channels_meta = []

    conn = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [fetch_channel(session, semaphore, u) for u in target_usernames]
        results = await asyncio.gather(*tasks)

        for idx, (title, posts, discovered) in enumerate(results):
            u = target_usernames[idx]
            if posts:
                all_new_posts.extend(posts)
                channels_meta.append({
                    "username": str(u),
                    "title": str(title or u),
                    "last_checked": datetime.utcnow().isoformat()
                })
            discovered_all.update(discovered)

    print(f"[+] Total new posts scraped: {len(all_new_posts)}")
    print(f"[+] Discovered new channels: {len(discovered_all)}")

    if all_new_posts:
        posts_df = pd.DataFrame(all_new_posts)
        posts_df["id"] = [str(int(datetime.utcnow().timestamp()) + i) for i in range(len(posts_df))]
        posts_out = os.path.join(OUTPUT_DIR, "p.parquet")
        posts_df.to_parquet(posts_out, compression="zstd", index=False)
        print(f"[+] Saved scraped posts to {posts_out}")

    c_records = channels_meta + [{"username": str(d), "title": str(d), "last_checked": ""} for d in discovered_all]
    if c_records:
        ch_df = pd.DataFrame(c_records).drop_duplicates(subset=["username"])
        ch_out = os.path.join(OUTPUT_DIR, "c.parquet")
        ch_df.to_parquet(ch_out, compression="zstd", index=False)
        print(f"[+] Saved channels to {ch_out}")

if __name__ == "__main__":
    asyncio.run(main())
