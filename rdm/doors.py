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
    eig_curve = []
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
        min_eig = float(w.min())
        eig_curve.append((float(sigma), min_eig))
        results.append({
            "sigma": float(sigma),
            "n": int(n),
            "min_eig": min_eig,
            "ok": ok,
        })
    # Monotonicity heuristic: min_eig should be non-decreasing as sigma decreases
    monotone = all(eig_curve[i+1][1] >= eig_curve[i][1] - 1e-12 for i in range(len(eig_curve)-1))
    return {"ok": ok_all, "details": results, "eig_curve": eig_curve, "monotone": bool(monotone)}


def certify_pick_on_interval_arctan(theta_fn, T1: float, T2: float, n: int = 256, sigma: float = 5e-3, tol: float = 1e-10) -> dict:
    """
    Alternate door: arctan-spaced nodes to emphasize endpoints.
    """
    k = np.arange(n)
    y = np.tan((k + 0.5) * (np.pi / (2 * n)) - np.pi / 4)
    t_grid = np.sort((T1 + T2) / 2.0 + (T2 - T1) * y)
    s_vals = 0.5 + sigma + 1j * t_grid
    theta_vals = np.array([theta_fn(complex(s - 0.5)) for s in s_vals])
    K = pick_kernel(theta_vals, s_vals)
    w = np.linalg.eigvalsh(K)
    return {"n": int(n), "sigma": float(sigma), "min_eig": float(w.min()), "ok": bool(w.min() >= -tol)}


