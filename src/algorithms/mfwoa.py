
import numpy as np
from ..fuzzy_entropy import sorted_thresholds

def mfwoa_optimize(task_objs, Ks, bounds=(1,254), pop=40, iters=100, rmp=0.3, seed=42):
    rng = np.random.default_rng(seed)
    maxK = max(Ks)
    lb, ub = bounds
    n_tasks = len(Ks)

    skill = rng.integers(0, n_tasks, size=pop)
    X = rng.uniform(lb, ub, size=(pop, maxK))
    fitness = np.full(pop, -np.inf, dtype=np.float64)
    best_per_task = [None]*n_tasks
    bestfit_per_task = [-np.inf]*n_tasks

    def eval_i(i):
        t = skill[i]
        K = Ks[t]
        f = task_objs[t]
        fit = f(X[i][:K])
        return fit

    for i in range(pop):
        fitness[i] = eval_i(i)
        t = skill[i]
        if fitness[i] > bestfit_per_task[t]:
            bestfit_per_task[t] = fitness[i]
            best_per_task[t] = X[i].copy()

    for it in range(iters):
        a = 2.0 - 2.0 * (it / max(1, iters-1))
        for i in range(pop):
            t = skill[i]
            K = Ks[t]
            best_t = best_per_task[t] if best_per_task[t] is not None else X[i]

            r1 = rng.random(maxK)
            r2 = rng.random(maxK)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2

            if rng.random() < rmp and n_tasks > 1:
                other_tasks = [j for j in range(n_tasks) if j != t and best_per_task[j] is not None]
                guide = best_per_task[int(rng.choice(other_tasks))] if other_tasks else best_t
            else:
                guide = best_t

            p = rng.random()
            if p < 0.5:
                if np.linalg.norm(A[:K], ord=2) >= 1:
                    j = int(rng.integers(0, pop))
                    X_rand = X[j]
                    D = np.abs(C[:K] * X_rand[:K] - X[i][:K])
                    new_vec = X_rand[:K] - A[:K] * D
                else:
                    D = np.abs(C[:K] * guide[:K] - X[i][:K])
                    new_vec = guide[:K] - A[:K] * D
            else:
                b = 1.0
                l = (rng.random(K) * 2.0) - 1.0
                D = np.abs(guide[:K] - X[i][:K])
                new_vec = D * np.exp(b * l) * np.cos(2*np.pi*l) + guide[:K]

            X_new = X[i].copy()
            X_new[:K] = np.clip(new_vec, lb, ub)
            old = X[i].copy()
            X[i] = X_new
            fit_new = eval_i(i)
            if fit_new < fitness[i]:
                X[i] = old
            else:
                fitness[i] = fit_new
                if fit_new > bestfit_per_task[t]:
                    bestfit_per_task[t] = fit_new
                    best_per_task[t] = X[i].copy()

    out_T, out_fit = [], []
    for ti, K in enumerate(Ks):
        if best_per_task[ti] is None:
            out_T.append(None); out_fit.append(float("-inf"))
        else:
            out_T.append(sorted_thresholds(best_per_task[ti], K))
            out_fit.append(float(bestfit_per_task[ti]))
    return out_T, out_fit
