# Deep Image Colorization Using U-Net in CIE Lab Color Space

The objective of this project is to build a deep-learning image colorization system where a grayscale image represented by the CIE Lab L* channel is provided as input to a custom U-Net, and the network predicts the a* and b* color channels. The predicted channels will later be combined with L* and converted back to RGB.

## Current Development Status
**Step 1 — Environment Setup**

## Basic Project Structure
```text
deep-image-colorization/
├── data/           # Datasets (raw, train, val, test)
├── models/         # Saved model checkpoints
├── outputs/        # Predictions and plots
├── src/            # Source code (dataset, model, training, inference)
├── scripts/        # Utility scripts (like setup verification)
├── requirements.txt# Project dependencies
└── README.md       # Project documentation
```

## Basic Environment Setup Instructions
1. Install Python 3.8 or higher.
2. Create and activate a virtual environment (optional but recommended).
3. Install PyTorch with CUDA support (e.g., for CUDA 12.6):
   ```bash
   pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
   ```
4. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Hardware Configuration
This project was primarily developed and tested with:
* **GPU**: NVIDIA GeForce RTX 3050
* **PyTorch**: 2.12.0 (cu126)

## Verify Setup
To verify that your environment is set up correctly, run the verification script:
```bash
python scripts/verify_setup.py
```
