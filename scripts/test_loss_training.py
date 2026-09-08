"""Test script for training pipeline integration and controlled learning diagnostic with Smooth L1 loss.

Covers:
Phase 5: Mini Training Pipeline Smoke Test (MPS, AMP, 5 train batches, 2 val batches, parameter updates).
Phase 6: Controlled Learning Diagnostic (single-batch overfitting for 30 iterations, gradient flow across all layers).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ColorizationUNet
from src.dataset import create_dataloader
from src.device import get_device, get_device_name, get_amp_device_type, get_memory_stats
from src.losses import get_loss_function, get_loss_metadata


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def run_mini_training_smoke_test(device, amp_type):
    print("==================================================")
    print("PHASE 5 — MINI TRAINING PIPELINE SMOKE TEST")
    print("==================================================")
    print(f"Device: {device} ({get_device_name()})")
    print(f"AMP Device Type: {amp_type}")

    train_manifest = os.path.join(PROJECT_ROOT, "data", "splits", "train.txt")
    val_manifest = os.path.join(PROJECT_ROOT, "data", "splits", "val.txt")

    train_loader = create_dataloader(train_manifest, batch_size=8, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_manifest, batch_size=8, shuffle=False, num_workers=0)

    model = ColorizationUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = get_loss_function("smooth_l1", beta=1.0)
    scaler = torch.amp.GradScaler(amp_type) if amp_type else None

    # Track parameter before training
    test_param = next(model.parameters())
    param_before = test_param.clone().detach()

    print("\n--- Training 5 Mini-Batches with Smooth L1 + AMP ---")
    model.train()
    train_losses = []
    for batch_idx, (l_batch, ab_batch) in enumerate(train_loader):
        if batch_idx >= 5:
            break
        l_batch = l_batch.to(device)
        ab_batch = ab_batch.to(device)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(amp_type):
                pred_ab = model(l_batch)
                loss = loss_fn(pred_ab, ab_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_ab = model(l_batch)
            loss = loss_fn(pred_ab, ab_batch)
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        assert not torch.isnan(loss).any(), f"NaN loss at batch {batch_idx}"
        assert not torch.isinf(loss).any(), f"Inf loss at batch {batch_idx}"
        assert loss_val > 0, f"Loss must be positive, got {loss_val}"
        train_losses.append(loss_val)
        print(f"  Batch {batch_idx + 1}/5 - Smooth L1 Loss: {loss_val:.6f}")

    # Verify parameter update
    param_after = test_param.clone().detach()
    param_diff = (param_after - param_before).abs().max().item()
    print(f"\nParameter Update Check (max |delta|): {param_diff:.6e}")
    assert param_diff > 0, "Model parameters did not update after optimizer.step()!"
    print("  [PASS] Weights successfully updated.")

    # Validation
    print("\n--- Validation 2 Mini-Batches with Smooth L1 ---")
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, (l_batch, ab_batch) in enumerate(val_loader):
            if batch_idx >= 2:
                break
            l_batch = l_batch.to(device)
            ab_batch = ab_batch.to(device)

            if amp_type:
                with torch.amp.autocast(amp_type):
                    pred_ab = model(l_batch)
                    v_loss = loss_fn(pred_ab, ab_batch)
            else:
                pred_ab = model(l_batch)
                v_loss = loss_fn(pred_ab, ab_batch)

            v_val = v_loss.item()
            assert not torch.isnan(v_loss).any(), f"NaN val loss at batch {batch_idx}"
            assert not torch.isinf(v_loss).any(), f"Inf val loss at batch {batch_idx}"
            assert v_val > 0, f"Val loss must be positive, got {v_val}"
            val_losses.append(v_val)
            print(f"  Val Batch {batch_idx + 1}/2 - Smooth L1 Val Loss: {v_val:.6f}")

    mem_stats = get_memory_stats(device)
    print(f"\nMemory stats: {mem_stats}")
    print("PHASE 5: ALL MINI-TRAINING CHECKS PASSED\n")


def run_controlled_learning_diagnostic(device, amp_type):
    print("==================================================")
    print("PHASE 6 — CONTROLLED LEARNING DIAGNOSTIC")
    print("==================================================")
    train_manifest = os.path.join(PROJECT_ROOT, "data", "splits", "train.txt")
    train_loader = create_dataloader(train_manifest, batch_size=8, shuffle=False, num_workers=0)

    # Grab a single fixed batch
    fixed_l, fixed_ab = next(iter(train_loader))
    fixed_l = fixed_l.to(device)
    fixed_ab = fixed_ab.to(device)

    print(f"Fixed batch loaded: L={list(fixed_l.shape)}, ab={list(fixed_ab.shape)}")

    # 1. Overfit with Smooth L1
    print("\n--- Diagnostic 1: Overfitting on Fixed Batch with Smooth L1 (30 iterations) ---")
    set_seed(42)
    model_sl1 = ColorizationUNet().to(device)
    optimizer_sl1 = optim.Adam(model_sl1.parameters(), lr=2e-4)
    loss_fn_sl1 = get_loss_function("smooth_l1", beta=1.0)
    scaler_sl1 = torch.amp.GradScaler(amp_type) if amp_type else None

    losses_sl1 = []
    for step in range(1, 31):
        model_sl1.train()
        optimizer_sl1.zero_grad()
        if scaler_sl1:
            with torch.amp.autocast(amp_type):
                pred = model_sl1(fixed_l)
                loss = loss_fn_sl1(pred, fixed_ab)
            scaler_sl1.scale(loss).backward()
            scaler_sl1.step(optimizer_sl1)
            scaler_sl1.update()
        else:
            pred = model_sl1(fixed_l)
            loss = loss_fn_sl1(pred, fixed_ab)
            loss.backward()
            optimizer_sl1.step()

        loss_val = loss.item()
        assert not torch.isnan(loss).any() and not torch.isinf(loss).any()
        losses_sl1.append(loss_val)

        if step == 1 or step % 5 == 0 or step == 30:
            print(f"  Step {step:2d}/30 | Smooth L1 Loss: {loss_val:.6f}")

    init_loss_sl1 = losses_sl1[0]
    final_loss_sl1 = losses_sl1[-1]
    reduction_sl1 = (init_loss_sl1 - final_loss_sl1) / init_loss_sl1 * 100.0

    print(f"\nInitial Loss: {init_loss_sl1:.6f}")
    print(f"Final Loss:   {final_loss_sl1:.6f}")
    print(f"Reduction:    {reduction_sl1:.2f}%")

    assert init_loss_sl1 > final_loss_sl1, "Smooth L1 loss failed to decrease!"
    assert reduction_sl1 >= 50.0, f"Expected at least 50% loss reduction, got {reduction_sl1:.2f}%"
    print("  [PASS] Smooth L1 successfully reduced loss by >50%.")

    # Verify gradient flow to all key submodules
    print("\n--- Gradient Flow Verification Across Architecture ---")
    # Do one backward pass without zero_grad to inspect grad tensors
    model_sl1.train()
    optimizer_sl1.zero_grad()
    pred = model_sl1(fixed_l)
    loss = loss_fn_sl1(pred, fixed_ab)
    loss.backward()

    # Check encoder, bottleneck, decoder
    grad_checks = {
        "Encoder (enc1)": model_sl1.enc1.block[0].weight.grad,
        "Bottleneck": model_sl1.bottleneck.block[0].weight.grad,
        "Decoder (dec1)": model_sl1.dec1.up.weight.grad,
        "Final Layer": model_sl1.final[0].weight.grad
    }

    for name, grad in grad_checks.items():
        assert grad is not None, f"Grad is None for {name}"
        assert not torch.isnan(grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(grad).any(), f"Inf gradient in {name}"
        max_g = grad.abs().max().item()
        mean_g = grad.abs().mean().item()
        assert max_g > 0, f"Zero gradient in {name}"
        print(f"  {name:18s} | Max |Grad|: {max_g:.6e} | Mean |Grad|: {mean_g:.6e} -> HEALTHY")

    print("  [PASS] Gradient flow confirmed through all architectural layers.")

    # 2. Informational comparison with MSE
    print("\n--- Informational Comparison: MSE on Identical Fixed Batch (30 iterations) ---")
    set_seed(42)
    model_mse = ColorizationUNet().to(device)
    optimizer_mse = optim.Adam(model_mse.parameters(), lr=2e-4)
    loss_fn_mse = get_loss_function("mse")
    scaler_mse = torch.amp.GradScaler(amp_type) if amp_type else None

    losses_mse = []
    for step in range(1, 31):
        model_mse.train()
        optimizer_mse.zero_grad()
        if scaler_mse:
            with torch.amp.autocast(amp_type):
                pred = model_mse(fixed_l)
                loss = loss_fn_mse(pred, fixed_ab)
            scaler_mse.scale(loss).backward()
            scaler_mse.step(optimizer_mse)
            scaler_mse.update()
        else:
            pred = model_mse(fixed_l)
            loss = loss_fn_mse(pred, fixed_ab)
            loss.backward()
            optimizer_mse.step()

        loss_val = loss.item()
        losses_mse.append(loss_val)
        if step == 1 or step % 5 == 0 or step == 30:
            print(f"  Step {step:2d}/30 | MSE Loss: {loss_val:.6f}")

    init_loss_mse = losses_mse[0]
    final_loss_mse = losses_mse[-1]
    reduction_mse = (init_loss_mse - final_loss_mse) / init_loss_mse * 100.0
    print(f"\nMSE Initial: {init_loss_mse:.6f} -> Final: {final_loss_mse:.6f} (Reduction: {reduction_mse:.2f}%)")
    print(f"Smooth L1 Initial: {init_loss_sl1:.6f} -> Final: {final_loss_sl1:.6f} (Reduction: {reduction_sl1:.2f}%)")
    print("PHASE 6: ALL CONTROLLED LEARNING CHECKS PASSED\n")


if __name__ == "__main__":
    device = get_device()
    amp_type = get_amp_device_type(device)
    set_seed(42)

    run_mini_training_smoke_test(device, amp_type)
    run_controlled_learning_diagnostic(device, amp_type)

    print("==================================================")
    print("ALL TRAINING PIPELINE AND LEARNING TESTS PASSED!")
    print("==================================================")
