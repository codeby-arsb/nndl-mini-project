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

## Dataset
We use the **ADE20K** dataset for this project. ADE20K provides a highly diverse collection of indoor, outdoor, natural, urban, and object scenes, which makes it an excellent general-purpose dataset for image colorization. 

* **Usage**: The images are strictly used as RGB source images. Semantic segmentation masks and object annotations provided by ADE20K are ignored.
* **Storage Strategy**: The raw dataset images are stored locally in `data/raw/` but are excluded from version control (Git). Instead, deterministic split manifests (`data/splits/*.txt`) containing relative paths are tracked in Git to ensure dataset reproducibility without inflating the repository size.
* **Initial Experimental Subset**:
  * Training: 10,000 images
  * Validation: 1,000 images
  * Test: 500 images
* **Random Seed**: 42

### Generating Splits and Inspecting
To automatically download the raw dataset (if not already present), run:
```bash
python scripts/download_dataset.py
```

Once the dataset is downloaded, generate the deterministic train/val/test splits:
```bash
python scripts/create_splits.py
```

To inspect the dataset statistics (resolutions, formats, splits count):
```bash
python scripts/inspect_dataset.py
```

## CIE Lab Preprocessing
This project converts RGB images into the CIE Lab color space to train the colorization network. 
* **Why Lab?** The CIE Lab color space separates lightness (L*) from color (a* and b*), allowing the U-Net to be trained solely on predicting color from a grayscale-like input without having to simultaneously predict brightness.
* **Inputs & Targets**: The L* channel is extracted, normalized to `[-1, 1]`, and fed as the single-channel input to the model. The a* and b* channels are normalized to `[-1, 1]` and serve as the two-channel prediction targets.
* **Resizing**: All images are uniformly resized to 256x256 using bilinear interpolation before conversion.
* **Reconstruction**: After predictions are made, the predicted a* and b* channels are combined with the original L* channel, denormalized, and converted back to standard RGB.

To verify the preprocessing pipeline, check numerical boundaries, and inspect the reconstruction accuracy, run:
```bash
python scripts/test_preprocessing.py
```
