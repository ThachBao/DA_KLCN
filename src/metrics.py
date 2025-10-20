
import numpy as np

def _one_hot(labels, num_classes):
    h, w = labels.shape
    oh = np.zeros((num_classes, h, w), dtype=np.uint8)
    for c in range(num_classes):
        oh[c] = (labels == c).astype(np.uint8)
    return oh

def dice_score(pred, gt, num_classes=None, eps=1e-7):
    if num_classes is None:
        num_classes = max(int(pred.max()), int(gt.max())) + 1
    P = _one_hot(pred, num_classes)
    G = _one_hot(gt, num_classes)
    dice_c = []
    for c in range(num_classes):
        inter = (P[c] & G[c]).sum()
        denom = P[c].sum() + G[c].sum()
        dice_c.append((2.0 * inter + eps) / (denom + eps))
    return float(np.mean(dice_c)), dice_c

def iou_score(pred, gt, num_classes=None, eps=1e-7):
    if num_classes is None:
        num_classes = max(int(pred.max()), int(gt.max())) + 1
    P = _one_hot(pred, num_classes)
    G = _one_hot(gt, num_classes)
    iou_c = []
    for c in range(num_classes):
        inter = (P[c] & G[c]).sum()
        union = P[c].sum() + G[c].sum() - inter
        iou_c.append((inter + eps) / (union + eps))
    return float(np.mean(iou_c)), iou_c

def psnr(img, ref, max_val=255.0, eps=1e-7):
    img = img.astype(np.float32)
    ref = ref.astype(np.float32)
    mse = np.mean((img - ref) ** 2)
    if mse <= 1e-20:
        return 99.0
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse + eps)

def ssim(img, ref, K=(0.01, 0.03), win_size=11):
    # Simplified SSIM using uniform filter (requires scipy)
    import numpy as np
    from scipy.ndimage import uniform_filter
    L = 255.0
    K1, K2 = K
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    img = img.astype(np.float32)
    ref = ref.astype(np.float32)
    mu1 = uniform_filter(img, win_size)
    mu2 = uniform_filter(ref, win_size)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = uniform_filter(img * img, win_size) - mu1_sq
    sigma2_sq = uniform_filter(ref * ref, win_size) - mu2_sq
    sigma12 = uniform_filter(img * ref, win_size) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-7)
    return float(np.mean(ssim_map))
