from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple


def poisson_outer_from_modulus(
    u: Callable[[float], float],
    sigma0: float,
    t_support: Tuple[float, float],
    samples: int = 2048,
) -> Callable[[complex], complex]:
    """
    Build an outer function O(s) on the half-plane {Re s > sigma0} from boundary modulus
    data u(t) = log|F(sigma0 + i t)| using the Poisson integral (numerical).

    Returns O(s) with |O(sigma0+it)| = exp(u(t)) approximately, assuming the integral
    over a finite window t_support and simple trapezoidal quadrature.
    """

    t1, t2 = t_support
    if t2 <= t1:
        raise ValueError("t_support must satisfy t2>t1")

    # Pre-sample boundary u on [t1,t2]
    dt = (t2 - t1) / (samples - 1)
    ts = [t1 + k * dt for k in range(samples)]
    us = [float(u(t)) for t in ts]

    def O(s: complex) -> complex:
        sigma = s.real
        t = s.imag
        if sigma <= sigma0:
            raise ValueError("O(s) is defined for Re s > sigma0")
        # Poisson kernel for half-plane at height sigma - sigma0
        a = sigma - sigma0
        acc = 0.0
        mass = 0.0
        for k, tau in enumerate(ts):
            kern = (1.0 / math.pi) * (a / (a * a + (t - tau) * (t - tau)))
            w = 0.5 if (k == 0 or k == samples - 1) else 1.0
            acc += w * kern * us[k]
            mass += w * kern
        acc *= dt
        mass *= dt
        if mass > 0:
            acc = acc / mass
        # Outer has modulus exp(P[u]) and is positive real (no phase here)
        return math.exp(acc)

    return O


