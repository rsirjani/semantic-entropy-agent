"""Download Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit from HuggingFace."""

import os
import sys

# Disable brotli to avoid httpx decompression errors
os.environ["HF_HUB_DISABLE_BROTLI"] = "1"

from huggingface_hub import snapshot_download

MODEL_REPO = "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
LOCAL_DIR = "D:/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"

def main():
    print(f"Downloading {MODEL_REPO} to {LOCAL_DIR}...")
    print("This will take ~30 minutes for ~18 GB of model files.")

    os.makedirs(LOCAL_DIR, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=LOCAL_DIR,
    )

    # Verify download
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(LOCAL_DIR):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
            file_count += 1

    print(f"\nDownload complete!")
    print(f"Files: {file_count}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Location: {LOCAL_DIR}")

if __name__ == "__main__":
    main()
