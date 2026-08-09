import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from utils import (
    load_rgb_image, resize_image, rgb_to_lab, normalize_lab, 
    denormalize_lab, lab_to_rgb, preprocess_image, reconstruct_rgb
)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    import math
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

test_split_path = os.path.join(project_root, "data", "splits", "test.txt")
with open(test_split_path, "r") as f:
    test_images = [line.strip() for line in f if line.strip()]

random.seed(42)
selected_images = random.sample(test_images, min(10, len(test_images)))

shape_test_pass = True
finite_test_pass = True
roundtrip_pass = True

maes = []
mses = []
psnrs = []

plot_images = []

print("==================================================")
print("PREPROCESSING VERIFICATION")
print("==================================================\n")
print(f"Images Tested: {len(selected_images)}\n")

for i, rel_path in enumerate(selected_images):
    img_path = os.path.join(project_root, rel_path)
    
    # 1. Load and resize
    raw_rgb = load_rgb_image(img_path)
    resized_rgb = resize_image(raw_rgb)
    resized_rgb_float = resized_rgb.astype(np.float32) / 255.0
    
    # 2. Lab conversion
    lab = rgb_to_lab(resized_rgb_float)
    L_raw = lab[:, :, 0]
    a_raw = lab[:, :, 1]
    b_raw = lab[:, :, 2]
    
    # 3. Normalize
    L_norm, ab_norm = normalize_lab(lab)
    
    # 4. Denormalize & Reconstruct
    lab_reconstructed = denormalize_lab(L_norm, ab_norm)
    reconstructed_rgb = lab_to_rgb(lab_reconstructed)
    
    # --- Verifications ---
    # Shape check
    if resized_rgb.shape != (256, 256, 3): shape_test_pass = False
    if L_norm.shape != (1, 256, 256): shape_test_pass = False
    if ab_norm.shape != (2, 256, 256): shape_test_pass = False
    if reconstructed_rgb.shape != (256, 256, 3): shape_test_pass = False
    
    # Finite value check
    if not (np.isfinite(resized_rgb_float).all() and 
            np.isfinite(L_raw).all() and 
            np.isfinite(a_raw).all() and 
            np.isfinite(b_raw).all() and 
            np.isfinite(L_norm).all() and 
            np.isfinite(ab_norm).all() and 
            np.isfinite(reconstructed_rgb).all()):
        finite_test_pass = False
        
    # Round-trip check (denormed lab vs original lab)
    lab_diff = np.max(np.abs(lab - lab_reconstructed))
    if lab_diff > 1e-3:
        roundtrip_pass = False
        
    # Accuracy check
    mae = np.mean(np.abs(resized_rgb_float - reconstructed_rgb))
    mse = np.mean((resized_rgb_float - reconstructed_rgb) ** 2)
    p = psnr(resized_rgb_float, reconstructed_rgb)
    
    maes.append(mae)
    mses.append(mse)
    psnrs.append(p)
    
    if i < 5:
        # Prepare for plotting
        L_vis = L_norm[0] # range [-1, 1]
        # Map to [0, 1] for visualization
        L_vis_mapped = (L_vis + 1.0) / 2.0
        plot_images.append({
            'original': resized_rgb_float,
            'l_channel': L_vis_mapped,
            'reconstructed': reconstructed_rgb
        })

print(f"Shape Test: {'PASS' if shape_test_pass else 'FAIL'}")
print(f"Finite Value Test: {'PASS' if finite_test_pass else 'FAIL'}")
print(f"Lab Normalization Round-Trip: {'PASS' if roundtrip_pass else 'FAIL'}\n")

print(f"Average MAE: {np.mean(maes):.6f}")
print(f"Average MSE: {np.mean(mses):.6f}")
print(f"Average PSNR: {np.mean(psnrs):.2f} dB\n")

# --- Visualization ---
plot_dir = os.path.join(project_root, "outputs", "plots")
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, "preprocessing_verification.png")

fig, axes = plt.subplots(len(plot_images), 3, figsize=(10, 3 * len(plot_images)))
if len(plot_images) == 1:
    axes = [axes]
    
for idx, data in enumerate(plot_images):
    ax_orig, ax_l, ax_recon = axes[idx]
    
    ax_orig.imshow(data['original'])
    ax_orig.set_title("Original RGB")
    ax_orig.axis('off')
    
    ax_l.imshow(data['l_channel'], cmap='gray', vmin=0, vmax=1)
    ax_l.set_title("L* Grayscale (Mapped [0,1])")
    ax_l.axis('off')
    
    ax_recon.imshow(data['reconstructed'])
    ax_recon.set_title("Reconstructed RGB")
    ax_recon.axis('off')

plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print("Verification Sheet:")
print(f"outputs/plots/preprocessing_verification.png\n")
print("==================================================")
status = "PASS" if (shape_test_pass and finite_test_pass and roundtrip_pass and np.mean(maes) < 0.05) else "FAIL"
print(f"PREPROCESSING STATUS: {status}")
print("==================================================")
