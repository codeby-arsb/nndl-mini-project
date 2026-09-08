"""Loss functions module for Deep Image Colorization.

Provides configurable loss functions for training:
- MSE Loss: Standard mean squared error regression loss (Baseline).
- Smooth L1 (Huber) Loss: Robust regression loss with quadratic behavior for small errors
  and linear behavior for large errors, reducing regression-to-the-mean penalty.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

SUPPORTED_LOSSES = ("mse", "smooth_l1")
DEFAULT_SMOOTH_L1_BETA = 1.0  # Standard PyTorch default beta for SmoothL1Loss

def get_loss_function(name: str = "mse", beta: float = DEFAULT_SMOOTH_L1_BETA) -> nn.Module:
    """Factory function to retrieve a configured loss function by name.

    Args:
        name: Name of the loss function ('mse' or 'smooth_l1').
        beta: Threshold parameter for Smooth L1 loss (default: 1.0).

    Returns:
        torch.nn.Module: Instantiated PyTorch loss function.

    Raises:
        ValueError: If an unsupported loss name is provided.
    """
    normalized_name = name.strip().lower()
    if normalized_name == "mse":
        return nn.MSELoss()
    elif normalized_name in ("smooth_l1", "smoothl1", "huber"):
        if beta <= 0:
            raise ValueError(f"beta parameter must be positive, got {beta}")
        return nn.SmoothL1Loss(beta=beta)
    else:
        raise ValueError(
            f"Unsupported loss function '{name}'. Supported losses are: {SUPPORTED_LOSSES}"
        )

def get_loss_metadata(name: str = "mse", beta: float = DEFAULT_SMOOTH_L1_BETA) -> Dict[str, Any]:
    """Retrieve metadata describing the loss function and configuration for logging/checkpoints.

    Args:
        name: Loss function name.
        beta: Beta threshold for Smooth L1 loss.

    Returns:
        dict: Metadata dictionary containing loss_name, loss_beta, and formulation details.
    """
    normalized_name = name.strip().lower()
    if normalized_name == "mse":
        return {
            "loss_name": "mse",
            "loss_beta": None,
            "formulation": "MSELoss (0.5 * (y_pred - y_true)^2)",
            "description": "Baseline Mean Squared Error regression loss."
        }
    elif normalized_name in ("smooth_l1", "smoothl1", "huber"):
        if beta <= 0:
            raise ValueError(f"beta parameter must be positive, got {beta}")
        return {
            "loss_name": "smooth_l1",
            "loss_beta": beta,
            "formulation": f"SmoothL1Loss(beta={beta}) (0.5*(x-y)^2/beta for |x-y|<beta, |x-y|-0.5*beta otherwise)",
            "description": f"Robust Smooth L1 / Huber regression loss with transition threshold beta={beta}."
        }
    else:
        raise ValueError(
            f"Unsupported loss function '{name}'. Supported losses are: {SUPPORTED_LOSSES}"
        )
