
import numpy as np

def _otsu_2class(hist):
    p = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(len(p)))
    mu_t = mu[-1]
    sigma_b2 = (mu_t*omega - mu)**2 / (omega*(1.0 - omega) + 1e-12)
    sigma_b2[omega*(1.0 - omega) < 1e-8] = 0.0
    t = int(np.argmax(sigma_b2))
    return t

def multi_otsu_thresholds(hist, classes=3):
    if classes <= 1:
        return np.array([], dtype=np.int32)
    try:
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(np.arange(256, dtype=np.uint8).repeat(hist.astype(int)), classes=classes)
        return thresholds.astype(np.int32)
    except Exception:
        regions = [(0, 255)]
        thresholds = []
        for _ in range(classes - 1):
            best_gain = -np.inf
            best_t = None
            best_region_idx = None
            for idx, (lo, hi) in enumerate(regions):
                h = hist[lo:hi+1].astype(np.float64)
                if h.sum() < 1e-9 or hi - lo < 2:
                    continue
                t_rel = _otsu_2class(h)
                t = lo + t_rel
                gain = h[:t_rel+1].sum() * h[t_rel+1:].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_t = t
                    best_region_idx = idx
            if best_t is None:
                break
            thresholds.append(best_t)
            lo, hi = regions.pop(best_region_idx)
            regions.extend([(lo, best_t), (best_t+1, hi)])
            regions.sort(key=lambda x: x[0])
        return np.array(sorted(thresholds), dtype=np.int32)
