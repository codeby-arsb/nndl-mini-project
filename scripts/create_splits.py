import os
import random
from PIL import Image

def verify_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(project_root, "data", "raw")
splits_dir = os.path.join(project_root, "data", "splits")
os.makedirs(splits_dir, exist_ok=True)

valid_exts = {".jpg", ".jpeg", ".png"}

print("Discovering images in data/raw...")
all_images = []
for root, dirs, files in os.walk(raw_dir):
    for f in files:
        if os.path.splitext(f)[1].lower() in valid_exts:
            # The ADE20K challenge data includes annotations. We skip them to keep only RGB images.
            if "annotations" not in root.lower():
                all_images.append(os.path.join(root, f))

print(f"Found {len(all_images)} potential RGB images.")
print("Verifying image integrity...")

valid_images = []
for img_path in all_images:
    if verify_image(img_path):
        rel_path = os.path.relpath(img_path, project_root)
        valid_images.append(rel_path.replace('\\', '/'))

print(f"Verified {len(valid_images)} valid images.")

# Deterministic sort
valid_images.sort()

# Deterministic shuffle
random.seed(42)
random.shuffle(valid_images)

train_count = 10000
val_count = 1000
test_count = 500
total_required = train_count + val_count + test_count

if len(valid_images) < total_required:
    print(f"WARNING: Not enough valid images. Have {len(valid_images)}, need {total_required}.")
    if len(valid_images) > 0:
        train_count = int(len(valid_images) * 0.8)
        val_count = int(len(valid_images) * 0.1)
        test_count = len(valid_images) - train_count - val_count
    else:
        train_count = 0
        val_count = 0
        test_count = 0

train_split = valid_images[:train_count]
val_split = valid_images[train_count:train_count+val_count]
test_split = valid_images[train_count+val_count:train_count+val_count+test_count]

def write_split(split_name, paths):
    if not paths:
        return
    split_path = os.path.join(splits_dir, f"{split_name}.txt")
    with open(split_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

write_split("train", train_split)
write_split("val", val_split)
write_split("test", test_split)

# Leakage check
train_set = set(train_split)
val_set = set(val_split)
test_set = set(test_split)

leak1 = len(train_set.intersection(val_set))
leak2 = len(train_set.intersection(test_set))
leak3 = len(val_set.intersection(test_set))

overlap = leak1 + leak2 + leak3

print(f"Train images: {len(train_split)}")
print(f"Validation images: {len(val_split)}")
print(f"Test images: {len(test_split)}")
print(f"Split overlap: {overlap}")
print(f"Corrupt selected images: 0") # Handled during selection
if overlap == 0 and len(valid_images) > 0:
    print("Dataset Leakage Check: PASS")
else:
    print("Dataset Leakage Check: FAIL")
