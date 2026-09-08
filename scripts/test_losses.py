"""Unit test suite for colorization loss functions.

Verifies:
1. Loss function creation and parameter validation.
2. Zero error verification.
3. Small error behavior (quadratic regime).
4. Large error behavior (linear regime & outlier robustness).
5. Gradient behavior (gradient saturation for Smooth L1 vs linear growth for MSE).
6. Device (MPS/CPU) and realistic tensor shape compatibility.
"""

import sys
import os
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.losses import get_loss_function, get_loss_metadata, SUPPORTED_LOSSES


def test_1_loss_creation():
    print("--- Test 1: Loss Function Creation ---")
    mse_loss = get_loss_function("mse")
    assert isinstance(mse_loss, nn.MSELoss), f"Expected MSELoss, got {type(mse_loss)}"
    print("  [PASS] MSE loss instantiated successfully.")

    smooth_l1_default = get_loss_function("smooth_l1")
    assert isinstance(smooth_l1_default, nn.SmoothL1Loss), f"Expected SmoothL1Loss, got {type(smooth_l1_default)}"
    assert smooth_l1_default.beta == 1.0, f"Expected default beta=1.0, got {smooth_l1_default.beta}"
    print("  [PASS] Smooth L1 (beta=1.0) instantiated successfully.")

    smooth_l1_custom = get_loss_function("smooth_l1", beta=0.5)
    assert smooth_l1_custom.beta == 0.5, f"Expected beta=0.5, got {smooth_l1_custom.beta}"
    print("  [PASS] Smooth L1 (beta=0.5) instantiated successfully.")

    try:
        get_loss_function("cross_entropy")
        assert False, "Expected ValueError for invalid loss name"
    except ValueError as e:
        print(f"  [PASS] Invalid loss name correctly raised ValueError: {e}")

    try:
        get_loss_function("smooth_l1", beta=-0.5)
        assert False, "Expected ValueError for negative beta"
    except ValueError as e:
        print(f"  [PASS] Negative beta correctly raised ValueError: {e}")

    print("Test 1: ALL CHECKS PASSED\n")


def test_2_zero_error():
    print("--- Test 2: Zero Error Verification ---")
    y_pred = torch.tensor([0.1, -0.4, 0.8, -0.2], dtype=torch.float32)
    y_true = torch.tensor([0.1, -0.4, 0.8, -0.2], dtype=torch.float32)

    mse_loss = get_loss_function("mse")
    smooth_l1_loss = get_loss_function("smooth_l1")

    loss_mse = mse_loss(y_pred, y_true).item()
    loss_smooth = smooth_l1_loss(y_pred, y_true).item()

    assert loss_mse == 0.0, f"MSE loss must be 0.0, got {loss_mse}"
    assert loss_smooth == 0.0, f"Smooth L1 loss must be 0.0, got {loss_smooth}"
    print(f"  [PASS] Zero error inputs yield loss = 0.0 (MSE: {loss_mse}, Smooth L1: {loss_smooth})")
    print("Test 2: ALL CHECKS PASSED\n")


def test_3_small_error_quadratic():
    print("--- Test 3: Small Error Behavior (Quadratic Regime |e| < beta) ---")
    beta = 1.0
    smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction="none")

    # Errors strictly smaller than beta
    errors = torch.tensor([0.05, 0.1, 0.25, 0.5, 0.8], dtype=torch.float32)
    y_true = torch.zeros_like(errors)
    y_pred = errors.clone()

    computed_loss = smooth_l1(y_pred, y_true)
    analytical_loss = 0.5 * (errors ** 2) / beta

    max_diff = torch.max(torch.abs(computed_loss - analytical_loss)).item()
    assert torch.allclose(computed_loss, analytical_loss, atol=1e-6), (
        f"Mismatch between computed and analytical quadratic loss: {max_diff}"
    )
    print(f"  [PASS] Verified quadratic regime formula (0.5 * e^2 / beta). Max diff: {max_diff:.2e}")
    print("Test 3: ALL CHECKS PASSED\n")


def test_4_large_error_linear():
    print("--- Test 4: Large Error Behavior (Linear Regime |e| >= beta) ---")
    beta = 1.0
    smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction="none")
    mse = nn.MSELoss(reduction="none")

    # Errors greater than or equal to beta
    errors = torch.tensor([1.0, 1.5, 2.0, 3.0, 5.0], dtype=torch.float32)
    y_true = torch.zeros_like(errors)
    y_pred = errors.clone()

    computed_smooth = smooth_l1(y_pred, y_true)
    analytical_smooth = errors.abs() - 0.5 * beta
    computed_mse = mse(y_pred, y_true)

    max_diff = torch.max(torch.abs(computed_smooth - analytical_smooth)).item()
    assert torch.allclose(computed_smooth, analytical_smooth, atol=1e-6), (
        f"Mismatch between computed and analytical linear loss: {max_diff}"
    )
    print(f"  [PASS] Verified linear regime formula (|e| - 0.5 * beta). Max diff: {max_diff:.2e}")

    # Robustness to outliers check: Smooth L1 must be strictly less than MSE for large errors
    assert (computed_smooth < computed_mse).all(), "Smooth L1 must be smaller than MSE for errors >= beta"
    for e, sl1, m in zip(errors.tolist(), computed_smooth.tolist(), computed_mse.tolist()):
        print(f"    Error {e:.1f}: Smooth L1 = {sl1:.4f} < MSE = {m:.4f} (Ratio: {sl1/m:.2%})")
    print("  [PASS] Verified Smooth L1 < MSE for large errors (robustness to outliers).")
    print("Test 4: ALL CHECKS PASSED\n")


def test_5_gradient_behavior():
    print("--- Test 5: Gradient Behavior & Saturation ---")
    beta = 1.0
    smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction="sum")
    mse = nn.MSELoss(reduction="sum")

    # Test single-element gradients across error range
    large_errors = [1.5, 2.5, 5.0, 10.0]
    for err in large_errors:
        # Smooth L1 gradient
        x_sl1 = torch.tensor([err], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([0.0], dtype=torch.float32)
        loss_sl1 = smooth_l1(x_sl1, target)
        loss_sl1.backward()
        grad_sl1 = x_sl1.grad.item()

        # MSE gradient
        x_mse = torch.tensor([err], dtype=torch.float32, requires_grad=True)
        loss_m = mse(x_mse, target)
        loss_m.backward()
        grad_mse = x_mse.grad.item()

        assert abs(grad_sl1 - 1.0) < 1e-5, f"Smooth L1 gradient should saturate to 1.0, got {grad_sl1}"
        assert abs(grad_mse - 2.0 * err) < 1e-5, f"MSE gradient should be 2*error={2*err}, got {grad_mse}"
        assert not torch.isnan(x_sl1.grad).any() and not torch.isinf(x_sl1.grad).any()
        assert not torch.isnan(x_mse.grad).any() and not torch.isinf(x_mse.grad).any()

        print(f"  Error {err:.1f}: Smooth L1 grad = {grad_sl1:.4f} (saturates to 1.0) | MSE grad = {grad_mse:.4f} (unbounded)")

    # Negative large errors
    x_neg = torch.tensor([-4.0], dtype=torch.float32, requires_grad=True)
    loss_neg = smooth_l1(x_neg, torch.tensor([0.0]))
    loss_neg.backward()
    assert abs(x_neg.grad.item() - (-1.0)) < 1e-5, f"Smooth L1 gradient for negative error should be -1.0, got {x_neg.grad.item()}"
    print(f"  Error -4.0: Smooth L1 grad = {x_neg.grad.item():.4f} (saturates to -1.0)")
    print("  [PASS] Gradient saturation confirmed; all gradients finite.")
    print("Test 5: ALL CHECKS PASSED\n")


def test_6_device_and_shape_compatibility():
    print("--- Test 6: Device and Shape Compatibility ---")
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    print(f"  Testing devices: {devices}")

    batch_sizes = [1, 4, 8]
    losses = {
        "mse": get_loss_function("mse"),
        "smooth_l1": get_loss_function("smooth_l1", beta=1.0)
    }

    for dev in devices:
        device = torch.device(dev)
        for loss_name, loss_fn in losses.items():
            for b in batch_sizes:
                # Shape: [B, 2, 256, 256] matching Lab 'ab' channels
                pred = torch.randn(b, 2, 256, 256, device=device, requires_grad=True)
                target = torch.randn(b, 2, 256, 256, device=device)

                loss = loss_fn(pred, target)
                assert loss.dim() == 0, f"Expected 0-dim scalar, got shape {loss.shape}"
                assert not torch.isnan(loss).any() and not torch.isinf(loss).any()

                # Backpropagation check
                loss.backward()
                assert pred.grad is not None
                assert pred.grad.shape == pred.shape
                assert not torch.isnan(pred.grad).any() and not torch.isinf(pred.grad).any()
                pred.grad.zero_()

            print(f"  [PASS] {loss_name.upper()} on {dev.upper()}: batches {batch_sizes} [B, 2, 256, 256] forward & backward pass successful.")

    print("Test 6: ALL CHECKS PASSED\n")


if __name__ == "__main__":
    print("==================================================")
    print("STANDALONE LOSS FUNCTION UNIT TEST SUITE")
    print("==================================================")
    test_1_loss_creation()
    test_2_zero_error()
    test_3_small_error_quadratic()
    test_4_large_error_linear()
    test_5_gradient_behavior()
    test_6_device_and_shape_compatibility()
    print("==================================================")
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
