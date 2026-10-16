import os
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from huggingface_hub import hf_hub_download

# ==================== إعدادات الكراولر ====================
CHANNELS_TO_SCRAPE_PER_RUN = 500   # عدد القنوات بكل دورة
CONCURRENT_REQUESTS = 10           # سرعة التوازي
OUTPUT_DIR = "crawler_output"
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def extract_channel_meta(soup, username):
    """استخراج كامل بيانات القناة بدقة"""
    title = username
    description = ""
    subscribers = "0"
    avatar_url = ""
    is_verified = False

    try:
        # العنوان
        title_el = soup.select_one(".tgme_channel_info_header_title span")
        if title_el:
            title = title_el.get_text(strip=True)

        # الوصف
        desc_el = soup.select_one(".tgme_channel_info_description")
        if desc_el:
            description = desc_el.get_text(separator="\n", strip=True)

        # المشتركون
        counters = soup.select(".tgme_channel_info_counter .counter_value")
        for c in counters:
            val = c.get_text(strip=True)
            if "subscribers" in c.parent.get_text().lower() or "members" in c.parent.get_text().lower():
                if "K" in val:
                    subscribers = str(int(float(val.replace("K", "")) * 1000))
                elif "M" in val:
                    subscribers = str(int(float(val.replace("M", "")) * 1000000))
                else:
                    subscribers = re.sub(r"[^\d]", "", val) or "0"

        # صورة البروفايل
        avatar_el = soup.select_one(".tgme_page_photo_image img")
        if avatar_el and avatar_el.get("src"):
            avatar_url = avatar_el["src"]

        # علامة التوثيق
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
        "last_checked": datetime.utcnow().isoformat()
    }

def extract_post_data(msg, username):
    """استخراج المنشور بكامل تفاصيله ومرفقاته"""
    try:
        post_div = msg.select_one(".tgme_widget_message")
        if not post_div:
            return None, set()

        data_post = post_div.get("data-post", "")
        if not data_post or "/" not in data_post:
            return None, set()

        post_id = data_post.split("/")[-1]
        unique_id = f"{username}_{post_id}"

        # النص والمنشنات
        text_el = msg.select_one(".tgme_widget_message_text")
        text = text_el.get_text(separator="\n", strip=True) if text_el else ""

        discovered = set()
        if text:
            mentions = re.findall(r"@([a-zA-Z0-9_]{4,32})", text)
            for m in mentions:
                discovered.add(m.lower())

        # الميديا (صور متعددة / فيديو / معاينة روابط)
        images = []
        video_url = ""
        media_type = "text"

        # صور البوست
        photo_thumbs = msg.select(".tgme_widget_message_photo_wrap")
        for thumb in photo_thumbs:
            style = thumb.get("style", "")
            match = re.search(r"background-image:url\('([^']+)'\)", style)
            if match:
                images.append(match.group(1))

        # فيديو
        video_el = msg.select_one("video")
        if video_el and video_el.get("src"):
            video_url = video_el["src"]
            media_type = "video"
        elif images:
            media_type = "image"

        primary_image = images[0] if images else ""

        # المشاهدات
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

        # التاريخ
        date_str = ""
        time_el = msg.select_one("time")
        if time_el and time_el.get("datetime"):
            date_str = time_el.get("datetime")

        # التفاعلات (Reactions) إن وجدت
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
            "image": str(primary_image),
            "all_images": ",".join(images) if len(images) > 1 else str(primary_image),
            "video": str(video_url),
            "media_type": str(media_type),
            "views": str(views),
            "reactions": str(reactions_count),
            "date": str(date_str)
        }, discovered

    except Exception:
        return None, set()

async def fetch_channel(session, semaphore, username):
    url = f"https://t.me/s/{username}"
    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=12) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    channel_meta = extract_channel_meta(soup, username)
                    posts = []
                    discovered_channels = set()

                    messages = soup.select(".tgme_widget_message_wrap")
                    for msg in messages:
                        post_data, found_ch = extract_post_data(msg, username)
                        if post_data:
                            posts.append(post_data)
                        discovered_channels.update(found_ch)

                    return channel_meta, posts, discovered_channels
        except Exception:
            pass
    return None, [], set()

async def main():
    print("[*] Starting Enhanced Telegram Cloud Scraper...")

    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_REPO_ID = os.getenv("HF_REPO_ID")
    target_usernames = []

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
            print(f"[+] Successfully sampled {len(target_usernames)} target channels!")
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

        for meta, posts, discovered in results:
            if meta:
                channels_meta.append(meta)
            if posts:
                all_new_posts.extend(posts)
            discovered_all.update(discovered)

    print(f"[+] Total posts scraped: {len(all_new_posts)}")
    print(f"[+] Updated channels meta: {len(channels_meta)}")
    print(f"[+] Discovered new channels: {len(discovered_all)}")

    # حفظ البوستات
    if all_new_posts:
        posts_df = pd.DataFrame(all_new_posts).drop_duplicates(subset=["id"], keep="last")
        posts_out = os.path.join(OUTPUT_DIR, "p.parquet")
        posts_df.to_parquet(posts_out, compression="zstd", index=False)
        print(f"[+] Saved enriched posts to {posts_out}")

    # حفظ القنوات المكتشفة والمحدثة
    new_channels_records = [{"username": str(d), "title": str(d), "description": "", "subscribers": "0", "avatar": "", "verified": "False", "last_checked": ""} for d in discovered_all]
    all_channels = channels_meta + new_channels_records
    if all_channels:
        ch_df = pd.DataFrame(all_channels).drop_duplicates(subset=["username"], keep="first")
        ch_out = os.path.join(OUTPUT_DIR, "c.parquet")
        ch_df.to_parquet(ch_out, compression="zstd", index=False)
        print(f"[+] Saved enriched channels to {ch_out}")

if __name__ == "__main__":
    asyncio.run(main())
