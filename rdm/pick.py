from __future__ import annotations

import numpy as np


def pick_kernel(theta_vals: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """
    Half-plane Pick kernel K_{jk} = (1 - θ(s_j) θ(s_k)^*) / (s_j + s_k^* - 1).
    Returns Hermitian symmetrized K.
    """
    n = len(s_vals)
    K = np.zeros((n, n), dtype=complex)
    for j in range(n):
        for k in range(n):
            num = 1.0 - theta_vals[j] * np.conjugate(theta_vals[k])
            den = s_vals[j] + np.conjugate(s_vals[k]) - 1.0
            K[j, k] = num / den
    return 0.5 * (K + K.conj().T)


def is_psd(K: np.ndarray, tol: float = 1e-12) -> bool:
    w = np.linalg.eigvalsh(K)
    return bool(w.min() >= -tol)


