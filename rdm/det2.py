from __future__ import annotations

import math
from typing import Iterable, Tuple


def det2_from_primes(s: complex, primes: Iterable[int]) -> complex:
    """
    det2(I - A_N(s)) for diagonal A_N(s) with entries p^{-s}.
    For a diagonal HS operator K with eigenvalues λ_n, det2(I-K) = Π (1-λ_n) exp(λ_n).
    """
    acc = 1.0 + 0.0j
    for p in primes:
        lam = complex(p) ** (-s)
        acc *= (1.0 - lam) * math.e ** (lam)
    return acc


def xi_completed(s: complex, primes: Tuple[int, ...] | None = None) -> complex:
    """
    Completed zeta ξ(s) = 1/2 s(1-s) π^{-s/2} Γ(s/2) ζ(s).
    Prefer mpmath for Γ and ζ; fallback to truncated Euler product if mpmath unavailable.
    """
    try:
        import mpmath as mp

        mp.dps = 50
        s_mp = mp.mpf(s.real) + 1j * mp.mpf(s.imag)
        pref = 0.5 * s_mp * (1 - s_mp) * mp.power(mp.pi, -s_mp / 2.0) * mp.gamma(s_mp / 2.0)
        zeta = mp.zeta(s_mp)
        val = pref * zeta
        return complex(val.real, val.imag)
    except Exception:
        # Truncated Euler product for ζ(s)
        if primes is None:
            raise
        zeta = 1.0 + 0.0j
        for p in primes:
            zeta *= 1.0 / (1.0 - complex(p) ** (-s))
        # crude Γ and π term via math for real parts only
        # we keep a rough surrogate sufficient for scaffolding
        gamma_approx = math.gamma(s.real / 2.0)
        pref = 0.5 * s * (1 - s) * math.pi ** (-s.real / 2.0) * gamma_approx
        return pref * zeta


def boundary_modulus_log(sigma0: float, eps: float, t: float, primes: Tuple[int, ...]) -> float:
    """
    u_ε(t) = log | det2(I - A_N(σ0+ε+it)) / ξ(σ0+ε+it) | at a boundary line.
    """
    s = complex(sigma0 + eps, t)
    d2 = det2_from_primes(s, primes)
    xi = xi_completed(s, primes)
    val = abs(d2) / (abs(xi) + 1e-30)
    # avoid log(0)
    return float(math.log(val + 1e-30))


def first_primes(n: int) -> Tuple[int, ...]:
    """
    Return the first n primes using a simple sieve with an upper bound n*(log n + log log n) + 3.
    Suitable for moderate n.
    """
    if n <= 0:
        return tuple()
    import math as _m
    if n < 6:
        bound = 15
    else:
        bound = int(n * (_m.log(n) + _m.log(_m.log(n))) + 3)
    sieve = bytearray(b"\x01") * (bound + 1)
    sieve[:2] = b"\x00\x00"
    p = 2
    while p * p <= bound:
        if sieve[p]:
            step = p
            start = p * p
            sieve[start: bound + 1: step] = b"\x00" * (((bound - start) // step) + 1)
        p += 1
    primes_list = [i for i, is_prime in enumerate(sieve) if is_prime]
    return tuple(primes_list[:n])

