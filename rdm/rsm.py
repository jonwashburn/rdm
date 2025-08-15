from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple


@dataclass(frozen=True)
class FiniteLedgerBlock:
    """
    Minimal placeholder for a finite ledger block used to define μ_RS.
    Each entry is a (tick_index, signed_cost_flux) at a chosen port.
    Costs are in units of the immutable generator δ>0; additivity is enforced by construction.
    """

    tick_fluxes: Tuple[Tuple[int, float], ...]  # (tick, flux) pairs

    def is_positive_measurable(self) -> bool:
        # μ_RS must be a positive measure: all contributions to the measure are ≥ 0.
        # Here we demand nonnegative aggregated mass per tick.
        from collections import defaultdict

        acc = defaultdict(float)
        for t, f in self.tick_fluxes:
            acc[t] += f
        return all(v >= 0.0 for v in acc.values())


def recognition_spectral_measure(block: FiniteLedgerBlock) -> Callable[[float], float]:
    """
    Construct a simple discrete μ_RS from a finite ledger block at a chosen port.
    Returns μ_RS as a function over t>=0 that integrates test functions by Riemann–Stieltjes sums.

    This is a minimal, canonical discrete form:
      μ_RS = Σ_k (mass_k) · δ(t - t_k), with mass_k ≥ 0 and t_k in nonnegative integers (ticks).
    """

    if not block.is_positive_measurable():
        raise ValueError("Block is not positive (aggregated flux per tick must be ≥ 0).")

    from collections import defaultdict

    mass = defaultdict(float)
    for t, f in block.tick_fluxes:
        if t < 0:
            raise ValueError("Tick indices must be nonnegative.")
        mass[t] += f

    support = sorted((int(t), float(v)) for t, v in mass.items() if v > 0.0)

    def mu(t: float) -> float:
        # cumulative measure μ_RS([0, t]) for t≥0 (right-continuous step function)
        if t < 0:
            return 0.0
        s = 0.0
        for tick, m in support:
            if tick <= t:
                s += m
        return s

    # attach support for inspection
    setattr(mu, "support", tuple(support))
    return mu


def laplace_admittance(mu: Callable[[float], float], s: complex, t_max: int | None = None) -> complex:
    """
    Φ_RS(s) ≈ ∫_0^∞ e^{-s t} dμ_RS(t) for Re s > 1/2.
    For a discrete μ_RS supported on integer ticks, the Stieltjes integral reduces to a finite sum
    if we bound the support by t_max or inspect mu.support.
    """

    if hasattr(mu, "support"):
        support: Iterable[Tuple[int, float]] = getattr(mu, "support")
    else:
        # Fallback: sample ticks up to t_max; user must provide bound.
        if t_max is None:
            raise ValueError("t_max required when mu.support is not available.")
        # Approximate discrete derivative by differences of the cumulative function
        support = []
        prev = 0.0
        for t in range(0, t_max + 1):
            cur = float(mu(t))
            dm = cur - prev
            if dm > 0:
                support.append((t, dm))
            prev = cur

    acc = 0.0 + 0.0j
    for t, m in support:
        acc += m * complex(pow(2.718281828459045, -s.real * t)) * complex(
            pow(2.718281828459045, -1j * s.imag * t)
        )
    return acc


def prime_window_block(primes: Tuple[int, ...], weight: str = "one") -> FiniteLedgerBlock:
    """
    Construct a simple prime-window finite ledger block for B2.PORT.
    Each prime contributes nonnegative flux at a single tick (tick=1) with weight:
      - "one": 1.0 per prime
      - "log": 1/log p per prime
    This is a canonical, positive, additive placeholder suitable for μ_RS prototyping.
    """
    fluxes = []
    for p in primes:
        if weight == "one":
            w = 1.0
        elif weight == "log":
            w = 1.0 / float(__import__("math").log(p))
        else:
            raise ValueError("Unknown weight; use 'one' or 'log'")
        fluxes.append((1, w))
    return FiniteLedgerBlock(tick_fluxes=tuple(fluxes))


