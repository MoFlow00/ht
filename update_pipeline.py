import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

api = HfApi(token=HF_TOKEN)

def sync_and_merge():
    data_dir = "data_sync"
    os.makedirs(data_dir, exist_ok=True)
    
    files = ["c.parquet", "p.parquet"]
    
    for filename in files:
        remote_path = None
        # 1. محاولة تنزيل الملف الحالي من Hugging Face إن وجد
        try:
            print(f"[*] Downloading current {filename} from Hugging Face...")
            remote_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=data_dir
            )
            current_df = pd.read_parquet(remote_path)
            print(f"[+] Loaded existing {filename} with {len(current_df):,} rows.")
        except Exception as e:
            print(f"[-] No existing file found or error: {e}. Creating new dataset.")
            current_df = pd.DataFrame()

        # 2. قراءة البيانات الجديدة المسحوبة من الكراولر
        new_file_path = os.path.join("crawler_output", filename)
        if not os.path.exists(new_file_path):
            print(f"[!] No new data found at {new_file_path}, skipping.")
            continue

        new_df = pd.read_parquet(new_file_path)
        print(f"[*] Merging {len(new_df):,} newly scraped rows into {filename}...")

        # 3. توحيد الأنواع لمنع أخطاء الـ Schema
        for col in new_df.columns:
            new_df[col] = new_df[col].astype("string")
        if not current_df.empty:
            for col in current_df.columns:
                current_df[col] = current_df[col].astype("string")

        # 4. دمج البيانات وإزالة التكرار
        if not current_df.empty:
            merged_df = pd.concat([current_df, new_df], ignore_index=True)
        else:
            merged_df = new_df

        # منع التكرار بناء على معرّف المنشور أو القناة
        if filename == "p.parquet" and "post_id" in merged_df.columns and "username" in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=["username", "post_id"], keep="last")
        elif filename == "c.parquet" and "username" in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=["username"], keep="last")

        final_out_path = os.path.join(data_dir, filename)
        merged_df.to_parquet(final_out_path, compression="zstd", index=False)
        print(f"[+] Successfully saved {filename} with total {len(merged_df):,} rows.")

        # 5. رفع الملف المحدث إلى Hugging Face
        print(f"[*] Uploading updated {filename} to Hugging Face...")
        api.upload_file(
            path_or_fileobj=final_out_path,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print(f"[✓] {filename} is now live and updated on Hugging Face!")

if __name__ == "__main__":
    sync_and_merge()
