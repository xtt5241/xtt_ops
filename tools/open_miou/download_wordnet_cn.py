import os
import sys
import time
import zipfile
import urllib.request
from urllib.error import URLError, HTTPError

# 目标安装目录（可通过命令行参数自定义）
DEFAULT_DIR = os.path.expanduser("/opt/data/private/xtt_OPS/OPS/tools/open_miou")

# 多镜像（按顺序尝试）
MIRRORS = [
    # jsDelivr（GitHub CDN）
    "https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/corpora/{fname}",
    # GitHub Raw（官方）
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{fname}",
    # GitHub 代理
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{fname}",
]

FILES = ["wordnet.zip", "omw-1.4.zip"]

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def download_with_progress(url, dst_path, timeout=30):
    """下载文件并显示进度条"""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total_size = resp.length
        if total_size is None:
            total_size = 0
        chunk_size = 8192
        downloaded = 0

        with open(dst_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    bar_len = 40
                    filled = int(bar_len * percent / 100)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(f"\r[{bar}] {percent:6.2f}% ({downloaded/1024:.1f} KB/{total_size/1024:.1f} KB)")
                    sys.stdout.flush()
            sys.stdout.write("\n")


def download_with_retries(urls, dst_path, max_retry_per_url=2):
    for url in urls:
        for attempt in range(1, max_retry_per_url + 1):
            try:
                print(f"[INFO] Downloading: {url}")
                download_with_progress(url, dst_path)
                print(f"[OK] Saved to: {dst_path}")
                return True
            except (HTTPError, URLError, TimeoutError) as e:
                print(f"[WARN] Failed ({attempt}/{max_retry_per_url}) -> {e}")
                time.sleep(1.2)
        print("[INFO] Try next mirror...")
    return False


def extract_zip(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"[OK] Extracted: {zip_path} -> {dest_dir}")
    os.remove(zip_path)


def ensure_corpora(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for fname in FILES:
        pkg_name = fname.replace(".zip", "")
        target_dir = os.path.join(base_dir, pkg_name)
        if os.path.exists(target_dir):
            print(f"[SKIP] {pkg_name} already exists: {target_dir}")
            continue

        urls = [pattern.format(fname=fname) for pattern in MIRRORS]
        tmp_zip = os.path.join(base_dir, fname)
        ok = download_with_retries(urls, tmp_zip)
        if not ok:
            print(f"[ERROR] All mirrors failed for {fname}.")
            sys.exit(2)
        extract_zip(tmp_zip, base_dir)


def print_hints(base_dir):
    print("\n✅ Done. NLTK 数据已就绪：")
    print(f"   {os.path.join(base_dir, 'wordnet')}")
    print(f"   {os.path.join(base_dir, 'omw-1.4')}")
    print("\n如果 NLTK 仍提示找不到数据，可以：")
    print("1) 在代码里显式添加路径：")
    print("   import nltk, os")
    print(f"   nltk.data.path.append(os.path.expanduser('{os.path.dirname(base_dir)}'))")
    print("2) 或设置环境变量：")
    print(f"   export NLTK_DATA={os.path.dirname(base_dir)}")


if __name__ == "__main__":
    base_dir = DEFAULT_DIR
    if len(sys.argv) >= 2:
        base_dir = os.path.abspath(sys.argv[1])
    ensure_corpora(base_dir)
    print_hints(base_dir)
