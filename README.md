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
3. Install dependencies using: `pip install -r requirements.txt`
4. If using a GPU, make sure you have the appropriate CUDA Toolkit installed along with the correct PyTorch wheels.

## Verify Setup
To verify that your environment is set up correctly, run the verification script:
```bash
python scripts/verify_setup.py
```
