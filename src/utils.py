
import os, json, random, numpy as np
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_gray(path, arr):
    from PIL import Image
    arr = np.clip(arr, 0, 255).astype('uint8')
    Image.fromarray(arr, mode='L').save(path)

def overlay_mask(img_gray, seg, alpha=0.5):
    """Simple overlay: normalize seg labels to 0..255 per class hue mapping."""
    import numpy as np, colorsys
    h, w = img_gray.shape
    num_classes = int(seg.max()) + 1
    colors = []
    for i in range(num_classes):
        hue = i / max(1, num_classes)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((int(255*rgb[0]), int(255*rgb[1]), int(255*rgb[2])))
    rgb_img = np.stack([img_gray]*3, axis=-1).astype(np.float32)
    color_mask = np.zeros_like(rgb_img)
    for c in range(num_classes):
        mask = (seg == c)
        color_mask[mask] = colors[c]
    out = (1 - alpha) * rgb_img + alpha * color_mask
    return out.astype(np.uint8)

def hungarian_match(pred, gt):
    """Map predicted labels to GT labels to maximize overlap (IoU)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_pred = int(pred.max()) + 1
    n_gt = int(gt.max()) + 1
    n = max(n_pred, n_gt)
    cost = np.ones((n, n), dtype=np.float64)
    for i in range(n_pred):
        Pi = (pred == i)
        for j in range(n_gt):
            Gj = (gt == j)
            inter = np.logical_and(Pi, Gj).sum()
            union = Pi.sum() + Gj.sum() - inter
            iou = inter / (union + 1e-9)
            cost[i, j] = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.arange(n, dtype=np.int32)
    perm[row_ind] = col_ind
    return perm
