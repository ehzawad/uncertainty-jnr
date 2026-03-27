"""Shared preprocessing utilities for jersey number recognition."""

import numpy as np
import cv2


def adaptive_resize(img: np.ndarray, target_size: tuple = (224, 224), max_ar: float = 2.0) -> np.ndarray:
    """Resize image to target size, handling extreme aspect ratios.

    For images with AR > max_ar (tall/thin like teamsnap 84x35):
    - Center-crops the height to bring AR within max_ar
    - Then resizes to target_size

    This prevents extreme horizontal stretching that makes digits unreadable.
    For normal AR images, behaves like standard cv2.resize.

    Args:
        img: (H, W, 3) uint8 image
        target_size: (height, width) target
        max_ar: maximum aspect ratio before cropping (default 2.0)

    Returns:
        Resized image at target_size
    """
    h, w = img.shape[:2]
    ar = h / max(w, 1)

    if ar > max_ar:
        # Tall/thin image — crop height to bring AR to max_ar
        # Keep the center vertical region (torso area where numbers are)
        new_h = int(w * max_ar)
        y_start = (h - new_h) // 2
        img = img[y_start:y_start + new_h, :]

    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)


def letterbox_resize(img: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Resize image to target size while preserving aspect ratio.

    Scales the image to fit within target_size, then pads the remaining
    area with the image's mean pixel value. This prevents digit distortion
    from squashing extreme aspect ratios (e.g., teamsnap 80x32 → 224x224).

    Standard in object detection (YOLO-style letterboxing).

    Args:
        img: (H, W, 3) uint8 image
        target_size: (height, width) target dimensions

    Returns:
        (target_h, target_w, 3) letterboxed image
    """
    h, w = img.shape[:2]
    th, tw = target_size

    # Already correct size
    if h == th and w == tw:
        return img

    # Scale to fit within target while preserving AR
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Create canvas with mean pixel value (reduces edge artifacts vs black)
    mean_pixel = img.mean(axis=(0, 1)).astype(np.uint8)
    canvas = np.full((th, tw, 3), mean_pixel, dtype=np.uint8)

    # Center the resized image on the canvas
    y_off = (th - new_h) // 2
    x_off = (tw - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas
