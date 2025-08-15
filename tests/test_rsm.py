from __future__ import annotations

import cmath

from rdm.rsm import FiniteLedgerBlock, recognition_spectral_measure, laplace_admittance


def test_mu_rs_toy():
    # toy: two ticks with flux 1 each
    block = FiniteLedgerBlock(tick_fluxes=((0, 1.0), (1, 1.0)))
    mu = recognition_spectral_measure(block)
    assert mu(0) >= 1.0 and mu(1) >= 2.0
    # Laplace at s=1 has value 1 + e^{-1}
    val = laplace_admittance(mu, 1.0 + 0.0j)
    assert abs(val - (1.0 + cmath.exp(-1.0))) < 1e-6


def test_mu_rs_prime_window():
    # simple proxy for a prime-window ledger: three ticks, nonnegative additivity
    block = FiniteLedgerBlock(tick_fluxes=((0, 0.5), (2, 0.75), (2, 0.25)))
    mu = recognition_spectral_measure(block)
    assert mu(0) >= 0.5 and mu(2) >= 1.5
    s = 1.0 + 0.0j
    val = laplace_admittance(mu, s)
    expected = 0.5 * cmath.exp(-s * 0) + 1.0 * cmath.exp(-s * 2)
    assert abs(val - expected) < 1e-6


