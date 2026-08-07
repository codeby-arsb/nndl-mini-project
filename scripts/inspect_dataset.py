import os
from PIL import Image
import collections

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(project_root, "data", "raw")
splits_dir = os.path.join(project_root, "data", "splits")

valid_exts = {".jpg", ".jpeg", ".png"}

all_images = []
for root, dirs, files in os.walk(raw_dir):
    for f in files:
        if os.path.splitext(f)[1].lower() in valid_exts:
            if "annotations" not in root.lower():
                all_images.append(os.path.join(root, f))

formats = collections.defaultdict(int)
widths = []
heights = []
portrait = 0
landscape = 0
square = 0
below_256 = 0
grayscale_count = 0
corrupt = 0

print("Inspecting all RGB images in data/raw...")

for path in all_images:
    try:
        with Image.open(path) as img:
            formats[img.format] += 1
            w, h = img.size
            widths.append(w)
            heights.append(h)
            if w < h:
                portrait += 1
            elif w > h:
                landscape += 1
            else:
                square += 1
                
            if w < 256 or h < 256:
                below_256 += 1
                
            if img.mode not in ('RGB', 'RGBA'):
                grayscale_count += 1
    except Exception:
        corrupt += 1

min_w = min(widths) if widths else 0
max_w = max(widths) if widths else 0
min_h = min(heights) if heights else 0
max_h = max(heights) if heights else 0
avg_w = sum(widths) / len(widths) if widths else 0
avg_h = sum(heights) / len(heights) if heights else 0

def load_split(name):
    path = os.path.join(splits_dir, f"{name}.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

train_split = load_split("train")
val_split = load_split("val")
test_split = load_split("test")

print("\n--- Image Statistics ---")
print(f"Total images discovered: {len(all_images)}")
print(f"Supported formats: {dict(formats)}")
print(f"Minimum width: {min_w}")
print(f"Maximum width: {max_w}")
print(f"Minimum height: {min_h}")
print(f"Maximum height: {max_h}")
print(f"Average width: {avg_w:.2f}")
print(f"Average height: {avg_h:.2f}")
print(f"Portrait image count: {portrait}")
print(f"Landscape image count: {landscape}")
print(f"Square image count: {square}")
print(f"Unreadable/corrupt image count: {corrupt}")
print(f"Grayscale/non-RGB image count: {grayscale_count}")
print(f"Images smaller than 256x256: {below_256}")

print(f"\n--- Splits ---")
print(f"Train split count: {len(train_split)}")
print(f"Validation split count: {len(val_split)}")
print(f"Test split count: {len(test_split)}")

train_set = set(train_split)
val_set = set(val_split)
test_set = set(test_split)
overlap = len(train_set.intersection(val_set)) + len(train_set.intersection(test_set)) + len(val_set.intersection(test_set))
print(f"Duplicate paths across splits: {overlap}")
