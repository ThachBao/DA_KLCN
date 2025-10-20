
import numpy as np
from ..fuzzy_entropy import sorted_thresholds

def ga_optimize(obj, K, bounds=(1,254), pop=30, iters=100, pc=0.9, pm=0.1, seed=42):
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    dim = K
    X = rng.uniform(lb, ub, size=(pop, dim))
    fitness = np.array([obj(x) for x in X], dtype=np.float64)

    def select_parent():
        i = rng.integers(0, pop)
        j = rng.integers(0, pop)
        return X[i] if fitness[i] > fitness[j] else X[j]

    for it in range(iters):
        new_pop = []
        while len(new_pop) < pop:
            if rng.random() < pc:
                p1 = select_parent().copy()
                p2 = select_parent().copy()
                alpha = 0.3
                low = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
                high = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
                c1 = rng.uniform(low, high)
                c2 = rng.uniform(low, high)
                new_pop.extend([c1, c2])
            else:
                new_pop.append(select_parent().copy())
        X = np.array(new_pop[:pop])
        mut_mask = rng.random(X.shape) < pm
        X[mut_mask] += rng.normal(0, (ub-lb)*0.05, size=mut_mask.sum())
        X = np.clip(X, lb, ub)
        fitness = np.array([obj(x) for x in X], dtype=np.float64)

    best_idx = int(np.argmax(fitness))
    return sorted_thresholds(X[best_idx], K), float(fitness[best_idx])
