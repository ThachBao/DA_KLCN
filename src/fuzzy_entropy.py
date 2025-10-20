
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sorted_thresholds(vec, K):
    v = np.clip(np.sort(np.round(vec[:K]).astype(np.int32)), 1, 254)
    for i in range(1, len(v)):
        if v[i] <= v[i-1]:
            v[i] = min(254, v[i-1] + 1)
    return v

def fuzzy_entropy_objective(hist, K, s=2.0):
    bins = np.arange(256, dtype=np.float32)

    def fe_score(x):
        T = sorted_thresholds(x, K)
        Kp1 = K + 1
        mu = np.zeros((Kp1, 256), dtype=np.float64)
        mu[0] = sigmoid((T[0] - bins) / s)
        mu[K] = sigmoid((bins - T[-1]) / s)
        for j in range(1, K):
            left = sigmoid((bins - T[j-1]) / s)
            right = sigmoid((T[j] - bins) / s)
            mu[j] = left * right
        denom = mu.sum(axis=0, keepdims=True) + 1e-12
        mu = mu / denom
        fe = -(mu * (np.log(mu + 1e-12))).sum(axis=0)
        score = float((fe * hist).sum())
        return score
    return fe_score

def apply_thresholds(img_gray, T, s=2.0):
    bins = np.arange(256, dtype=np.float32)
    K = len(T)
    Kp1 = K + 1
    mu = np.zeros((Kp1, 256), dtype=np.float64)
    mu[0] = 1.0 / (1.0 + np.exp(- (T[0] - bins) / s))
    mu[K] = 1.0 / (1.0 + np.exp(- (bins - T[-1]) / s))
    for j in range(1, K):
        left = 1.0 / (1.0 + np.exp(- (bins - T[j-1]) / s))
        right = 1.0 / (1.0 + np.exp(- (T[j] - bins) / s))
        mu[j] = left * right
    denom = mu.sum(axis=0, keepdims=True) + 1e-12
    mu = mu / denom
    lut = mu.argmax(axis=0).astype(np.uint8)
    return lut[img_gray]
