import sys
import os

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"  {name} version: {version}")
        return module
    except ImportError:
        print(f"  {name} version: NOT INSTALLED")
        return None

print(f"1. Python version: {sys.version.split(' ')[0]}")

print("2.", end="")
torch = check_package("PyTorch", "torch")

print("3.", end="")
torchvision = check_package("Torchvision", "torchvision")

if torch is not None:
    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    print(f"4. CUDA availability: {cuda_avail}")
    print(f"   MPS built: {mps_built}")
    print(f"   MPS availability: {mps_avail}")
    print(f"   CPU availability: True")
    if cuda_avail:
        print(f"5. CUDA version detected by PyTorch: {torch.version.cuda}")
        print(f"6. Number of CUDA-capable GPUs detected: {torch.cuda.device_count()}")
        print(f"7. GPU name(s): {torch.cuda.get_device_name(0)}")
        device = "cuda"
    elif mps_avail:
        print("5. CUDA version detected by PyTorch: N/A (Apple Silicon MPS active)")
        print("6. Number of CUDA-capable GPUs detected: 0 (Using MPS accelerator)")
        print("7. GPU name(s): Apple Silicon GPU (MPS)")
        device = "mps"
    else:
        print("5. CUDA version detected by PyTorch: N/A")
        print("6. Number of CUDA-capable GPUs detected: 0")
        print("7. GPU name(s): N/A")
        device = "cpu"
    print(f"8. Selected computation device: {device}")
else:
    print("4. CUDA availability: N/A")
    print("   MPS built: N/A")
    print("   MPS availability: N/A")
    print("   CPU availability: N/A")
    print("5. CUDA version detected by PyTorch: N/A")
    print("6. Number of CUDA-capable GPUs detected: N/A")
    print("7. GPU name(s): N/A")
    print("8. Selected computation device: N/A")
    device = None

print("9.", end="")
np = check_package("NumPy", "numpy")

print("10.", end="")
cv2 = check_package("OpenCV", "cv2")

print("11.", end="")
skimage = check_package("scikit-image", "skimage")

print("12.", end="")
matplotlib = check_package("Matplotlib", "matplotlib")

print("\n--- Basic PyTorch Test ---")
if torch is not None:
    try:
        x = torch.rand(3, 3).to(device)
        y = torch.rand(3, 3).to(device)
        z = x + y
        print("PyTorch tensor test: PASSED")
    except Exception as e:
        print(f"PyTorch tensor test: FAILED - {e}")
else:
    print("PyTorch tensor test: FAILED (PyTorch not installed)")

print("\n--- Project Directory Verification ---")
dirs = [
    "data/raw",
    "data/train",
    "data/val",
    "data/test",
    "models",
    "outputs/checkpoints",
    "outputs/predictions",
    "outputs/plots",
    "src",
    "scripts"
]

all_dirs_exist = True
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for d in dirs:
    dir_path = os.path.join(project_root, os.path.normpath(d))
    if os.path.isdir(dir_path):
        print(f"Directory {d}: EXISTS")
    else:
        print(f"Directory {d}: MISSING")
        all_dirs_exist = False

print(f"\nDirectory verification: {'PASSED' if all_dirs_exist else 'FAILED'}")
