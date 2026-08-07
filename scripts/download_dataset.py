import urllib.request
import zipfile
import os

url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(project_root, "data", "raw")
zip_path = os.path.join(raw_dir, "ADEChallengeData2016.zip")

print("Downloading ADE20K dataset (ADEChallengeData2016) (~900MB)...")
try:
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
    print("Extraction complete. Cleaning up...")
    os.remove(zip_path)
    print("Done.")
except Exception as e:
    print(f"Failed: {e}")
