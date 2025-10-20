
import numpy as np
from ..fuzzy_entropy import sorted_thresholds

def woa_optimize(obj, K, bounds=(1,254), pop=20, iters=100, seed=42):
    rng = np.random.default_rng(seed)
    dim = K
    lb, ub = bounds
    X = rng.uniform(lb, ub, size=(pop, dim))
    fitness = np.array([obj(x) for x in X], dtype=np.float64)
    best_idx = int(np.argmax(fitness))
    best = X[best_idx].copy()
    best_fit = float(fitness[best_idx])

    for t in range(iters):
        a = 2.0 - 2.0 * (t / max(1, iters-1))
        for i in range(pop):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = rng.random()
            if p < 0.5:
                if np.linalg.norm(A, ord=2) >= 1:
                    rand_idx = rng.integers(0, pop)
                    X_rand = X[rand_idx]
                    D = np.abs(C * X_rand - X[i])
                    X_new = X_rand - A * D
                else:
                    D = np.abs(C * best - X[i])
                    X_new = best - A * D
            else:
                b = 1.0
                l = (rng.random(dim) * 2.0) - 1.0
                D = np.abs(best - X[i])
                X_new = D * np.exp(b * l) * np.cos(2*np.pi*l) + best

            X_new = np.clip(X_new, lb, ub)
            fit_new = obj(X_new)
            if fit_new > fitness[i]:
                X[i] = X_new
                fitness[i] = fit_new
                if fit_new > best_fit:
                    best_fit = fit_new
                    best = X_new.copy()

    return sorted_thresholds(best, K), best_fit
