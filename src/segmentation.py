
import numpy as np
from .fuzzy_entropy import apply_thresholds

def apply_thresholds_to_image(img_gray, T, mode='fuzzy', s=2.0):
    if mode == 'fuzzy':
        return apply_thresholds(img_gray, T, s=s)
    else:
        K = len(T)
        seg = np.zeros_like(img_gray, dtype=np.uint8)
        prev = -1
        for k, t in enumerate(T):
            seg[(img_gray > prev) & (img_gray <= t)] = k
            prev = t
        seg[img_gray > prev] = K
        return seg
