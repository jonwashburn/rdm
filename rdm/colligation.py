from __future__ import annotations

import numpy as np


def prime_grid_lossless(primes: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Canonical diagonal lossless colligation tied to primes (prototype):
    A = -diag(2/log p), P = I, C = sqrt(2*diag(2/log p)), D = -I, B = C.
    Satisfies lossless equalities: A^*P+PA = -C^*C, PB = -C^*D, D^*D = I.
    """
    lam = np.array([2.0 / np.log(float(p)) for p in primes], dtype=float)
    A = -np.diag(lam)
    P = np.eye(len(primes))
    C = np.diag(np.sqrt(2.0 * lam))
    D = -np.eye(len(primes))
    B = C.copy()
    return A, B, C, D, P


def kyp_lossless_equalities(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, P: np.ndarray, atol: float = 1e-10) -> bool:
    """
    Verify the lossless equalities:
      A^*P + P A = -C^* C
      P B = - C^* D
      D^* D = I
    """
    lhs1 = A.T @ P + P @ A
    rhs1 = -C.T @ C
    lhs2 = P @ B
    rhs2 = -C.T @ D
    lhs3 = D.T @ D
    rhs3 = np.eye(D.shape[0])

    ok1 = np.allclose(lhs1, rhs1, atol=atol)
    ok2 = np.allclose(lhs2, rhs2, atol=atol)
    ok3 = np.allclose(lhs3, rhs3, atol=atol)
    return bool(ok1 and ok2 and ok3)


def transfer_H(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, s: complex) -> np.ndarray:
    """
    H(s) = D + C (s I - A)^{-1} B  on the right half-plane.
    """
    n = A.shape[0]
    M = (s * np.eye(n) - A)
    X = np.linalg.solve(M, B)
    return D + C @ X


def scalar_port(H: np.ndarray, u: np.ndarray, v: np.ndarray) -> complex:
    """
    Extract scalar port h(s) = v^* H(s) u for unit vectors u,v.
    """
    return complex(v.conj().T @ H @ u)


