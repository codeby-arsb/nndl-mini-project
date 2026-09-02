import urllib.request
import zipfile
import os
import sys
import time

url = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(project_root, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)
zip_path = os.path.join(raw_dir, "ADEChallengeData2016.zip")

print("Downloading ADE20K dataset (ADEChallengeData2016) (~900MB)...")

def download_with_resume(url, filepath):
    max_retries = 10
    total_size = None
    
    # Get total size
    req = urllib.request.Request(url, method='HEAD')
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get('Content-Length', 0))
    except Exception as e:
        print(f"Warning: Could not get Content-Length: {e}")

    for attempt in range(max_retries):
        downloaded = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        if total_size and downloaded >= total_size:
            print("Download already completed.")
            break
            
        headers = {}
        if downloaded > 0:
            headers['Range'] = f"bytes={downloaded}-"
            mode = 'ab'
            print(f"Resuming download from byte {downloaded} (attempt {attempt+1}/{max_retries})...")
        else:
            mode = 'wb'
            print(f"Starting download (attempt {attempt+1}/{max_retries})...")
            
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(filepath, mode) as f:
                chunk_size = 1024 * 1024  # 1MB
                last_print = time.time()
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_print > 2:
                        last_print = time.time()
                        if total_size:
                            pct = (downloaded / total_size) * 100
                            mb_down = downloaded / (1024 * 1024)
                            mb_tot = total_size / (1024 * 1024)
                            print(f"  {mb_down:.1f} / {mb_tot:.1f} MB ({pct:.1f}%)")
                        else:
                            mb_down = downloaded / (1024 * 1024)
                            print(f"  {mb_down:.1f} MB downloaded")
            
            # Check if complete
            if total_size is None or os.path.getsize(filepath) >= total_size:
                print("Download complete.")
                break
        except Exception as e:
            print(f"Connection interrupted: {e}. Retrying in 2s...")
            time.sleep(2)

try:
    download_with_resume(url, zip_path)
    
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Downloaded file is not a valid zip archive.")
        
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
        
    print("Extraction complete. Cleaning up...")
    os.remove(zip_path)
    print("Done.")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
