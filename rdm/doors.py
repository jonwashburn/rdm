from __future__ import annotations

import numpy as np
from .pick import pick_kernel, is_psd


def certify_pick_on_interval(theta_fn, T1: float, T2: float, sigmas=(1e-2, 5e-3, 2e-3, 1e-3), ns=(64, 96, 128, 192), tol: float = 1e-10) -> dict:
    """
    Door D6: Build half-plane Pick matrices on grids approaching the boundary and certify PSD.
    Returns a dict with min eigenvalues and a boolean 'ok' per schedule.
    """
    results = []
    ok_all = True
    for sigma, n in zip(sigmas, ns):
        # Chebyshev-like nodes
        k = np.arange(n)
        x = np.cos(np.pi * (2 * k + 1) / (2 * n))
        t_grid = np.sort((T1 + T2) / 2.0 + (T2 - T1) * x / 2.0)
        s_vals = 0.5 + sigma + 1j * t_grid
        theta_vals = np.array([theta_fn(complex(s - 0.5)) for s in s_vals])
        K = pick_kernel(theta_vals, s_vals)
        w = np.linalg.eigvalsh(K)
        ok = bool(w.min() >= -tol)
        ok_all = ok_all and ok
        results.append({
            "sigma": float(sigma),
            "n": int(n),
            "min_eig": float(w.min()),
            "ok": ok,
        })
    return {"ok": ok_all, "details": results}


