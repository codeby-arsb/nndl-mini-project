import os
import math
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# Adjust path for imports if needed
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import load_rgb_image, resize_image, rgb_to_lab, normalize_lab, reconstruct_rgb

def get_default_paths():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(project_root, "outputs", "evaluation")
    plots_dir = os.path.join(project_root, "outputs", "plots")
    fixed_test_file = os.path.join(eval_dir, "fixed_test_images.txt")
    return project_root, eval_dir, plots_dir, fixed_test_file

def load_fixed_test_images(fixed_test_file: Optional[str] = None) -> List[str]:
    """Load the deterministic list of 10 test image relative paths."""
    if fixed_test_file is None:
        _, _, _, fixed_test_file = get_default_paths()
        
    if not os.path.exists(fixed_test_file):
        raise FileNotFoundError(f"Fixed test images list not found: {fixed_test_file}")
        
    with open(fixed_test_file, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
        
    if len(paths) != 10:
        raise ValueError(f"Expected exactly 10 fixed test images, found {len(paths)}")
        
    return paths

def compute_rgb_metrics(gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> Tuple[float, float, float]:
    """Compute RGB MAE, MSE, and PSNR between ground truth and predicted RGB images."""
    mae = float(np.mean(np.abs(gt_rgb - pred_rgb)))
    mse = float(np.mean((gt_rgb - pred_rgb) ** 2))
    psnr = float(20.0 * math.log10(1.0 / math.sqrt(mse))) if mse > 0 else 100.0
    return mae, mse, psnr

def evaluate_fixed_test_images(
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_name: str,
    epoch_num: Optional[int] = None,
    amp_type: Optional[str] = None,
    fixed_test_file: Optional[str] = None,
    plots_dir: Optional[str] = None,
    eval_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run qualitative and quantitative evaluation on the 10 fixed test images.
    Saves visual comparison sheet and appends metrics to CSVs.
    """
    project_root, default_eval_dir, default_plots_dir, default_fixed_file = get_default_paths()
    eval_dir = eval_dir or default_eval_dir
    plots_dir = plots_dir or default_plots_dir
    fixed_test_file = fixed_test_file or default_fixed_file
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    test_paths = load_fixed_test_images(fixed_test_file)
    
    model.eval()
    
    records = []
    plot_items = []
    
    for rel_path in test_paths:
        full_path = os.path.join(project_root, rel_path)
        image_id = os.path.basename(rel_path)
        
        raw_rgb = load_rgb_image(full_path)
        resized_rgb = resize_image(raw_rgb)
        orig_rgb_float = resized_rgb.astype(np.float32) / 255.0
        
        lab = rgb_to_lab(orig_rgb_float)
        L_norm, ab_norm = normalize_lab(lab)
        gt_rgb = reconstruct_rgb(L_norm, ab_norm)
        
        # Inference
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float().to(device)
        with torch.no_grad():
            if amp_type:
                with torch.amp.autocast(amp_type):
                    pred_ab_tensor = model(L_tensor)
            else:
                pred_ab_tensor = model(L_tensor)
                
        pred_ab_np = pred_ab_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        pred_rgb = reconstruct_rgb(L_norm, pred_ab_np)
        
        mae, mse, psnr = compute_rgb_metrics(gt_rgb, pred_rgb)
        
        records.append({
            "checkpoint": checkpoint_name,
            "image_id": image_id,
            "mae": mae,
            "mse": mse,
            "psnr": psnr
        })
        
        L_vis = (L_norm[0] + 1.0) / 2.0  # mapped from [-1, 1] to [0, 1] for visualization
        plot_items.append({
            "image_id": image_id,
            "original": orig_rgb_float,
            "input_l": L_vis,
            "ground_truth": gt_rgb,
            "prediction": pred_rgb,
            "mae": mae,
            "psnr": psnr
        })
        
    mean_mae = float(np.mean([r["mae"] for r in records]))
    mean_mse = float(np.mean([r["mse"] for r in records]))
    mean_psnr = float(np.mean([r["psnr"] for r in records]))
    
    # 1. Append per-image metrics to CSV
    per_image_csv = os.path.join(eval_dir, "test_metrics_per_image.csv")
    write_header = not os.path.exists(per_image_csv)
    with open(per_image_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["checkpoint", "image_id", "mae", "mse", "psnr"])
        for r in records:
            writer.writerow([r["checkpoint"], r["image_id"], f"{r['mae']:.6f}", f"{r['mse']:.6f}", f"{r['psnr']:.2f}"])
            
    # 2. Append summary metrics to CSV
    summary_csv = os.path.join(eval_dir, "test_metrics_summary.csv")
    write_summary_header = not os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_summary_header:
            writer.writerow(["checkpoint", "mean_mae", "mean_mse", "mean_psnr"])
        writer.writerow([checkpoint_name, f"{mean_mae:.6f}", f"{mean_mse:.6f}", f"{mean_psnr:.2f}"])
        
    # 3. Create and save visual comparison sheet
    plot_filename = f"predictions_{checkpoint_name}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    
    fig, axes = plt.subplots(len(plot_items), 4, figsize=(14, 3 * len(plot_items)))
    title_epoch_str = f"Epoch {epoch_num}" if epoch_num is not None else checkpoint_name
    fig.suptitle(f"Fixed Test Set Evaluation — Checkpoint: {checkpoint_name} ({title_epoch_str})\n"
                 f"Mean MAE: {mean_mae:.4f} | Mean MSE: {mean_mse:.5f} | Mean PSNR: {mean_psnr:.2f} dB",
                 fontsize=14, fontweight="bold", y=0.995)
                 
    col_titles = ["Original", "Input L*", "Ground Truth", "Prediction"]
    
    for row_idx, item in enumerate(plot_items):
        ax_orig = axes[row_idx, 0]
        ax_l = axes[row_idx, 1]
        ax_gt = axes[row_idx, 2]
        ax_pred = axes[row_idx, 3]
        
        ax_orig.imshow(np.clip(item["original"], 0.0, 1.0))
        ax_l.imshow(item["input_l"], cmap="gray", vmin=0.0, vmax=1.0)
        ax_gt.imshow(np.clip(item["ground_truth"], 0.0, 1.0))
        ax_pred.imshow(np.clip(item["prediction"], 0.0, 1.0))
        
        if row_idx == 0:
            for c_idx, title in enumerate(col_titles):
                axes[row_idx, c_idx].set_title(title, fontsize=12, fontweight="bold")
                
        ax_orig.set_ylabel(f"{item['image_id'][:16]}\nPSNR: {item['psnr']:.1f}dB", fontsize=9)
        
        for ax in (ax_orig, ax_l, ax_gt, ax_pred):
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"[{checkpoint_name}] Fixed test evaluation complete: Mean MAE={mean_mae:.4f}, MSE={mean_mse:.5f}, PSNR={mean_psnr:.2f} dB -> {plot_filename}")
    
    return {
        "checkpoint": checkpoint_name,
        "mean_mae": mean_mae,
        "mean_mse": mean_mse,
        "mean_psnr": mean_psnr,
        "records": records,
        "plot_path": plot_path
    }
