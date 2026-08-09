import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb

def load_rgb_image(path: str) -> np.ndarray:
    """Load an image from the given path as an RGB numpy array."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image at {path}. File may be missing or corrupted.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image(image: np.ndarray, size: tuple = (256, 256)) -> np.ndarray:
    """Resize the image to the specified size using bilinear interpolation."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image (uint8 or float) to CIE Lab color space."""
    if rgb_image.dtype == np.uint8:
        rgb_float = rgb_image.astype(np.float32) / 255.0
    else:
        rgb_float = rgb_image
    return rgb2lab(rgb_float)

def normalize_lab(lab_image: np.ndarray) -> tuple:
    """
    Extract L and ab channels from Lab image, normalize them, 
    and return as channel-first float32 arrays.
    """
    L = lab_image[:, :, 0]
    ab = lab_image[:, :, 1:]
    
    L_normalized = L / 50.0 - 1.0
    ab_normalized = ab / 128.0
    
    L_out = np.expand_dims(L_normalized, axis=0).astype(np.float32)
    ab_out = np.transpose(ab_normalized, (2, 0, 1)).astype(np.float32)
    
    return L_out, ab_out

def denormalize_lab(L_normalized: np.ndarray, ab_normalized: np.ndarray) -> np.ndarray:
    """
    Denormalize channel-first L and ab arrays and combine into a channel-last Lab image.
    """
    L = L_normalized[0]
    ab = np.transpose(ab_normalized, (1, 2, 0))
    
    L_denorm = (L + 1.0) * 50.0
    ab_denorm = ab * 128.0
    
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float64)
    lab[:, :, 0] = L_denorm
    lab[:, :, 1:] = ab_denorm
    return lab

def lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """Convert a CIE Lab image back to RGB [0, 1]."""
    rgb = lab2rgb(lab_image)
    return np.clip(rgb, 0.0, 1.0)

def preprocess_image(image: np.ndarray) -> tuple:
    """Full preprocessing pipeline from raw RGB to normalized L and ab channels."""
    resized = resize_image(image)
    lab = rgb_to_lab(resized)
    L_out, ab_out = normalize_lab(lab)
    return L_out, ab_out

def reconstruct_rgb(L_normalized: np.ndarray, ab_normalized: np.ndarray) -> np.ndarray:
    """Full reconstruction pipeline from normalized L and ab channels to RGB image."""
    lab = denormalize_lab(L_normalized, ab_normalized)
    rgb = lab_to_rgb(lab)
    return rgb
