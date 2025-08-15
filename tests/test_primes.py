from __future__ import annotations

from rdm.det2 import first_primes


def test_first_primes_small():
    assert first_primes(0) == tuple()
    assert first_primes(1) == (2,)
    assert first_primes(5) == (2, 3, 5, 7, 11)

