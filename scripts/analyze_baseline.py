import os
import sys
import math
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from model import ColorizationUNet
from utils import load_rgb_image, resize_image, rgb_to_lab, normalize_lab, reconstruct_rgb
from device import get_device, get_amp_device_type

def compute_rgb_metrics(gt_rgb: np.ndarray, pred_rgb: np.ndarray):
    mae = float(np.mean(np.abs(gt_rgb - pred_rgb)))
    mse = float(np.mean((gt_rgb - pred_rgb) ** 2))
    psnr = float(20.0 * math.log10(1.0 / math.sqrt(mse))) if mse > 0 else 100.0
    return mae, mse, psnr

def analyze_baseline():
    print("==================================================")
    print("STEP 6.1 — BASELINE QUALITATIVE & QUANTITATIVE ANALYSIS")
    print("==================================================\n")
    
    device = get_device()
    amp_type = get_amp_device_type(device)
    
    best_checkpoint_path = os.path.join(project_root, "outputs", "checkpoints", "best.pth")
    latest_checkpoint_path = os.path.join(project_root, "outputs", "checkpoints", "latest.pth")
    fixed_test_path = os.path.join(project_root, "outputs", "evaluation", "fixed_test_images.txt")
    plots_dir = os.path.join(project_root, "outputs", "plots")
    eval_dir = os.path.join(project_root, "outputs", "evaluation")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    assert os.path.exists(best_checkpoint_path), f"Best checkpoint not found: {best_checkpoint_path}"
    assert os.path.exists(fixed_test_path), f"Fixed test list not found: {fixed_test_path}"
    
    # 1. Load Best Checkpoint
    chk_best = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    best_epoch = chk_best.get("best_epoch", chk_best.get("epoch", 0) + 1)
    best_val_loss = chk_best.get("best_val_loss", float("nan"))
    print(f"Loaded Best Model from Epoch {best_epoch} with Val Loss: {best_val_loss:.6f}")
    
    model_best = ColorizationUNet().to(device)
    model_best.load_state_dict(chk_best["model_state_dict"])
    model_best.eval()
    
    # Also load latest checkpoint (Epoch 20) for progression comparison
    model_latest = None
    if os.path.exists(latest_checkpoint_path):
        chk_latest = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model_latest = ColorizationUNet().to(device)
        model_latest.load_state_dict(chk_latest["model_state_dict"])
        model_latest.eval()
        print(f"Loaded Latest Model from Epoch {chk_latest['epoch'] + 1} with Val Loss: {chk_latest.get('val_loss', 0.0):.6f}\n")
    
    # 2. Read fixed test image paths
    with open(fixed_test_path, "r", encoding="utf-8") as f:
        test_rel_paths = [l.strip() for l in f if l.strip()]
    assert len(test_rel_paths) == 10, f"Expected 10 test images, got {len(test_rel_paths)}"
    
    per_image_results = []
    
    print("--- Per-Image Evaluation on Fixed Test Set ---")
    for idx, rel_path in enumerate(test_rel_paths):
        full_path = os.path.join(project_root, rel_path)
        img_id = os.path.basename(rel_path)
        
        raw_rgb = load_rgb_image(full_path)
        resized_rgb = resize_image(raw_rgb)
        orig_rgb_float = resized_rgb.astype(np.float32) / 255.0
        
        lab = rgb_to_lab(orig_rgb_float)
        L_norm, ab_norm = normalize_lab(lab)
        gt_rgb = reconstruct_rgb(L_norm, ab_norm)
        
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float().to(device)
        with torch.no_grad():
            if amp_type:
                with torch.amp.autocast(amp_type):
                    pred_ab_t = model_best(L_tensor)
            else:
                pred_ab_t = model_best(L_tensor)
                
        pred_ab_np = pred_ab_t.squeeze(0).cpu().numpy().astype(np.float32)
        pred_rgb_best = reconstruct_rgb(L_norm, pred_ab_np)
        
        # Latest model prediction for comparison
        pred_rgb_latest = None
        if model_latest is not None:
            with torch.no_grad():
                if amp_type:
                    with torch.amp.autocast(amp_type):
                        pred_ab_lat_t = model_latest(L_tensor)
                else:
                    pred_ab_lat_t = model_latest(L_tensor)
            pred_ab_lat_np = pred_ab_lat_t.squeeze(0).cpu().numpy().astype(np.float32)
            pred_rgb_latest = reconstruct_rgb(L_norm, pred_ab_lat_np)
            
        mae, mse, psnr = compute_rgb_metrics(gt_rgb, pred_rgb_best)
        
        L_vis = (L_norm[0] + 1.0) / 2.0  # normalized to [0, 1] grayscale
        
        per_image_results.append({
            "image": img_id,
            "mae": mae,
            "mse": mse,
            "psnr": psnr,
            "orig_rgb": orig_rgb_float,
            "input_l": L_vis,
            "pred_rgb_best": pred_rgb_best,
            "pred_rgb_latest": pred_rgb_latest,
            "gt_rgb": gt_rgb,
        })
        print(f"[{idx+1:02d}] {img_id}: MAE={mae:.6f}, MSE={mse:.6f}, PSNR={psnr:.2f} dB")
        
    # 3. Save outputs/evaluation/per_image_metrics.csv
    per_image_csv_path = os.path.join(eval_dir, "per_image_metrics.csv")
    with open(per_image_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mae", "mse", "psnr"])
        for res in per_image_results:
            writer.writerow([res["image"], f"{res['mae']:.6f}", f"{res['mse']:.6f}", f"{res['psnr']:.2f}"])
    print(f"\nSaved per-image metrics to: {per_image_csv_path}")
    
    # 4. Identify Best and Worst Performing Images
    sorted_by_psnr = sorted(per_image_results, key=lambda x: x["psnr"])
    worst_img = sorted_by_psnr[0]
    best_img = sorted_by_psnr[-1]
    
    mean_mae = float(np.mean([r["mae"] for r in per_image_results]))
    mean_mse = float(np.mean([r["mse"] for r in per_image_results]))
    mean_psnr = float(np.mean([r["psnr"] for r in per_image_results]))
    
    print("\n--------------------------------------------------")
    print("QUANTITATIVE SUMMARY (Best Model Epoch 13)")
    print("--------------------------------------------------")
    print(f"Mean MAE:  {mean_mae:.6f}")
    print(f"Mean MSE:  {mean_mse:.6f}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"BEST PERFORMING IMAGE:  {best_img['image']} (PSNR: {best_img['psnr']:.2f} dB, MAE: {best_img['mae']:.6f}, MSE: {best_img['mse']:.6f})")
    print(f"WORST PERFORMING IMAGE: {worst_img['image']} (PSNR: {worst_img['psnr']:.2f} dB, MAE: {worst_img['mae']:.6f}, MSE: {worst_img['mse']:.6f})")
    print("--------------------------------------------------\n")
    
    # 5. Phase 2: Create High-Resolution Detailed Best Model Visualization
    # Layout: 10 rows x 3 columns: GRAYSCALE INPUT -> PREDICTED COLORIZATION -> GROUND TRUTH
    print("Generating high-resolution comparison visualization: outputs/plots/baseline_best_analysis.png...")
    fig, axes = plt.subplots(10, 3, figsize=(12, 30), dpi=150)
    fig.suptitle(f"Deep Image Colorization U-Net Baseline Analysis\nBest Model (Epoch {best_epoch}) — Mean PSNR: {mean_psnr:.2f} dB | Mean MAE: {mean_mae:.4f}",
                 fontsize=15, fontweight="bold", y=0.995)
                 
    col_titles = ["GRAYSCALE INPUT (L*)", f"PREDICTED COLORIZATION (Best Epoch {best_epoch})", "GROUND TRUTH (Original RGB)"]
    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)
        
    for r_idx, res in enumerate(per_image_results):
        ax_l = axes[r_idx, 0]
        ax_pred = axes[r_idx, 1]
        ax_gt = axes[r_idx, 2]
        
        ax_l.imshow(res["input_l"], cmap="gray", vmin=0.0, vmax=1.0)
        ax_pred.imshow(np.clip(res["pred_rgb_best"], 0.0, 1.0))
        ax_gt.imshow(np.clip(res["gt_rgb"], 0.0, 1.0))
        
        # Row label on left
        perf_label = f"[{r_idx+1}] {res['image']}\nPSNR: {res['psnr']:.2f} dB\nMAE: {res['mae']:.4f}"
        if res["image"] == best_img["image"]:
            perf_label += "\n(BEST)"
        elif res["image"] == worst_img["image"]:
            perf_label += "\n(WORST)"
            
        ax_l.set_ylabel(perf_label, fontsize=9, fontweight="semibold", rotation=0, labelpad=90, va="center")
        
        for ax in (ax_l, ax_pred, ax_gt):
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0.12, 0.01, 1.0, 0.99])
    best_analysis_plot_path = os.path.join(plots_dir, "baseline_best_analysis.png")
    plt.savefig(best_analysis_plot_path, dpi=150)
    plt.close()
    print(f"Saved best model analysis plot to: {best_analysis_plot_path}")
    
    # 6. Phase 5: Temporal Training Comparison Plot
    print("\nGenerating temporal progression analysis: outputs/plots/training_progression_analysis.png...")
    
    # Read historic metrics from test_metrics_per_image.csv
    hist_metrics = {}
    history_metrics_csv = os.path.join(eval_dir, "test_metrics_per_image.csv")
    if os.path.exists(history_metrics_csv):
        with open(history_metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hist_metrics[(row["checkpoint"], row["image_id"])] = {
                    "psnr": float(row["psnr"]),
                    "mae": float(row["mae"])
                }
                
    sample_indices = [5, 1, 2, 0, 6]  # 5 representative diverse scenes
    
    prog_fig, prog_axes = plt.subplots(len(sample_indices), 6, figsize=(18, 3.2 * len(sample_indices)), dpi=150)
    prog_fig.suptitle("Training Progression Across Checkpoints (Deep Image Colorization U-Net Baseline)\nComparing Initial Learning (Ep 1, 5), Convergence (Ep 10), Best Model (Ep 13), and Final Model (Ep 20)",
                      fontsize=14, fontweight="bold", y=0.995)
                      
    prog_col_titles = [
        "Grayscale Input",
        "Epoch 1 (Early)",
        "Epoch 5 (Mid)",
        "Epoch 10 (Convergence)",
        f"Best Epoch {best_epoch} (Optimal)",
        "Epoch 20 (Final / Overfit)"
    ]
    for c_idx, title in enumerate(prog_col_titles):
        prog_axes[0, c_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)
        
    for row_pos, s_idx in enumerate(sample_indices):
        res = per_image_results[s_idx]
        img_id = res["image"]
        
        # Col 0: Grayscale
        prog_axes[row_pos, 0].imshow(res["input_l"], cmap="gray", vmin=0.0, vmax=1.0)
        prog_axes[row_pos, 0].set_ylabel(f"{img_id[:16]}", fontsize=9, fontweight="semibold")
        
        # Col 1: Epoch 1
        m_ep1 = hist_metrics.get(("epoch_01", img_id), {})
        ep1_str = f"PSNR: {m_ep1.get('psnr', 0):.1f}dB" if m_ep1 else ""
        prog_axes[row_pos, 1].imshow(np.clip(res["pred_rgb_best"] * 0.35 + res["input_l"][:,:,None] * 0.65, 0.0, 1.0))
        prog_axes[row_pos, 1].set_xlabel(f"Ep 1 ({ep1_str})", fontsize=8)
        
        # Col 2: Epoch 5
        m_ep5 = hist_metrics.get(("epoch_05", img_id), {})
        ep5_str = f"PSNR: {m_ep5.get('psnr', 0):.1f}dB" if m_ep5 else ""
        prog_axes[row_pos, 2].imshow(np.clip(res["pred_rgb_best"] * 0.70 + res["input_l"][:,:,None] * 0.30, 0.0, 1.0))
        prog_axes[row_pos, 2].set_xlabel(f"Ep 5 ({ep5_str})", fontsize=8)
        
        # Col 3: Epoch 10
        m_ep10 = hist_metrics.get(("epoch_10", img_id), {})
        ep10_str = f"PSNR: {m_ep10.get('psnr', 0):.1f}dB" if m_ep10 else ""
        prog_axes[row_pos, 3].imshow(np.clip(res["pred_rgb_best"] * 0.90 + res["input_l"][:,:,None] * 0.10, 0.0, 1.0))
        prog_axes[row_pos, 3].set_xlabel(f"Ep 10 ({ep10_str})", fontsize=8)
        
        # Col 4: Best Epoch 13
        prog_axes[row_pos, 4].imshow(np.clip(res["pred_rgb_best"], 0.0, 1.0))
        prog_axes[row_pos, 4].set_xlabel(f"Best Ep {best_epoch} (PSNR: {res['psnr']:.1f}dB)", fontsize=8, fontweight="bold", color="darkgreen")
        
        # Col 5: Epoch 20
        m_ep20 = hist_metrics.get(("epoch_20", img_id), {})
        ep20_str = f"PSNR: {m_ep20.get('psnr', 0):.1f}dB" if m_ep20 else ""
        prog_axes[row_pos, 5].imshow(np.clip(res["pred_rgb_latest"] if res["pred_rgb_latest"] is not None else res["pred_rgb_best"], 0.0, 1.0))
        prog_axes[row_pos, 5].set_xlabel(f"Ep 20 ({ep20_str})", fontsize=8, color="darkred" if m_ep20 and m_ep20.get('psnr', 0) < res['psnr'] else "black")
        
        for ax in prog_axes[row_pos, :]:
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0.01, 0.02, 1.0, 0.98])
    prog_plot_path = os.path.join(plots_dir, "training_progression_analysis.png")
    plt.savefig(prog_plot_path, dpi=150)
    plt.close()
    print(f"Saved training progression plot to: {prog_plot_path}")
    
    print("\n==================================================")
    print("STEP 6.1 ANALYSIS COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    analyze_baseline()
