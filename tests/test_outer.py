from __future__ import annotations

import math
from rdm.outer import poisson_outer_from_modulus


def test_poisson_outer_constant_modulus():
    # If u(t)=c is constant, Poisson integral returns O(s)=exp(c) everywhere
    c = 0.7
    u = lambda t: c
    O = poisson_outer_from_modulus(u, sigma0=0.5, t_support=(0.0, 10.0), samples=256)
    val = O(0.6 + 0.0j)
    assert abs(val - math.exp(c)) < 5e-3

