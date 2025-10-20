
import numpy as np
from ..fuzzy_entropy import sorted_thresholds

def pso_optimize(obj, K, bounds=(1,254), pop=20, iters=100, w=0.72, c1=1.49, c2=1.49, seed=42):
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    dim = K
    X = rng.uniform(lb, ub, size=(pop, dim))
    V = rng.normal(0, (ub-lb)*0.1, size=(pop, dim))
    pbest = X.copy()
    pbest_fit = np.array([obj(x) for x in X], dtype=np.float64)

    g_idx = int(np.argmax(pbest_fit))
    gbest = pbest[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    for it in range(iters):
        r1 = rng.random((pop, dim))
        r2 = rng.random((pop, dim))
        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
        X = np.clip(X + V, lb, ub)
        fit = np.array([obj(x) for x in X], dtype=np.float64)
        better = fit > pbest_fit
        pbest[better] = X[better]
        pbest_fit[better] = fit[better]
        if float(fit.max()) > gbest_fit:
            g_idx = int(np.argmax(fit))
            gbest = X[g_idx].copy()
            gbest_fit = float(fit[g_idx])

    return sorted_thresholds(gbest, K), gbest_fit
