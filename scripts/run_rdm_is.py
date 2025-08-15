#!/usr/bin/env python3
from __future__ import annotations

import cmath
import numpy as np

from rdm.colligation import prime_grid_lossless, kyp_lossless_equalities, transfer_H, scalar_port
from rdm.pick import pick_kernel, is_psd
from rdm.rsm import FiniteLedgerBlock, recognition_spectral_measure, laplace_admittance, prime_window_block
from rdm.outer import poisson_outer_from_modulus
from rdm.doors import certify_pick_on_interval
from rdm.det2 import boundary_modulus_log
import json
import pathlib
import hashlib
import os
import sys
import time
import random


def main() -> None:
    # Deterministic seed for reproducibility
    seed = 123456789
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    # Toy prime grid and unit port
    primes = [2, 3, 5]
    A, B, C, D, P = prime_grid_lossless(primes)
    assert kyp_lossless_equalities(A, B, C, D, P), "Lossless KYP equalities failed"

    # Build scalar Schur θ(s) = H(s) (lossless) at a few nodes near the boundary
    def theta(s: complex) -> complex:
        H = transfer_H(A, B, C, D, s)
        # single-port extraction (first basis vector)
        e1 = np.zeros((len(primes), 1), dtype=complex)
        e1[0, 0] = 1.0
        h = scalar_port(H, e1, e1)
        # Use h directly as Schur (|h|<=1)
        return h

    sigma = 1e-2
    t_grid = np.linspace(0.0, 10.0, 64)
    s_vals = 0.5 + sigma + 1j * t_grid
    # Shift to z-domain (Re z>0) for the lossless transfer; evaluate θ(s)=θ_z(s-1/2)
    theta_vals = np.array([theta(complex(s - 0.5)) for s in s_vals])
    K = pick_kernel(theta_vals, s_vals)
    ok = is_psd(K, tol=1e-10)
    print(f"Pick PSD on toy grid: {ok}; min eig={np.linalg.eigvalsh(K).min():.3e}")

    # Artifact saving to per-run stamped directory
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_dir = pathlib.Path("artifacts") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "grid_s_vals.npy", s_vals)
    np.save(out_dir / "grid_theta_vals.npy", theta_vals)
    np.save(out_dir / "grid_pick_K.npy", K)

    # μ_RS toy test + outer normalization demo (artifacts preview)
    block = FiniteLedgerBlock(tick_fluxes=((0, 1.0), (1, 1.0)))
    mu = recognition_spectral_measure(block)
    def u_eps_demo(t: float) -> float:
        # demo: boundary modulus log|Φ_RS(1/2+ε+it)| with ε=0.01
        s = 0.5 + 0.01 + 1j * float(t)
        val = laplace_admittance(mu, s, t_max=2)
        return float(np.log(abs(val) + 1e-16))

    O = poisson_outer_from_modulus(u_eps_demo, sigma0=0.5, t_support=(0.0, 4.0), samples=512)
    o_val = O(0.6 + 0.0j)
    print(f"Outer demo O(0.6+0i)≈{o_val:.6f}")
    with open(out_dir / "outer_demo.json", "w") as f:
        json.dump({"O(0.6+0i)": float(o_val)}, f, indent=2)

    # D6.PICK certification on multiple intervals (export log)
    intervals = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    certs = []
    all_ok = True
    for (T1, T2) in intervals:
        cert = certify_pick_on_interval(theta, T1=T1, T2=T2)
        cert["interval"] = [float(T1), float(T2)]
        certs.append(cert)
        all_ok = all_ok and bool(cert["ok"])
    with open(out_dir / "pick_certification.json", "w") as f:
        json.dump({"ok": all_ok, "certs": certs}, f, indent=2)
    print(f"Pick certification overall ok={all_ok}")

    # B2.PORT μ_RS from a prime-window block (integrated)
    pw_block = prime_window_block(tuple(primes), weight="log")
    mu_pw = recognition_spectral_measure(pw_block)
    s = 0.5 + 0.05 + 0j
    phi_pw = laplace_admittance(mu_pw, s)
    with open(out_dir / "mu_rs_prime_window.json", "w") as f:
        json.dump({"s": [s.real, s.imag], "phi_pw": [phi_pw.real, phi_pw.imag], "support": getattr(mu_pw, "support")}, f, indent=2)

    # B3.OUTER (boundary modulus) from det2/xi along Re s = 1/2+ε
    eps = 0.01
    t_grid_outer = np.linspace(0.0, 20.0, 200)
    u_vals = [boundary_modulus_log(0.5, eps, float(t), tuple(primes)) for t in t_grid_outer]
    with open(out_dir / "outer_boundary_modulus.json", "w") as f:
        json.dump({"sigma0": 0.5, "eps": eps, "t": t_grid_outer.tolist(), "u": u_vals}, f, indent=2)

    # Simple manifest: env, versions, hashes, seeds, timestamps
    def sha256(path: pathlib.Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "seed": seed,
        "intervals": intervals,
        "artifacts": {
            "grid_s_vals.npy": sha256(out_dir / "grid_s_vals.npy"),
            "grid_theta_vals.npy": sha256(out_dir / "grid_theta_vals.npy"),
            "grid_pick_K.npy": sha256(out_dir / "grid_pick_K.npy"),
            "outer_demo.json": sha256(out_dir / "outer_demo.json"),
            "pick_certification.json": sha256(out_dir / "pick_certification.json"),
            "mu_rs_prime_window.json": sha256(out_dir / "mu_rs_prime_window.json"),
            "outer_boundary_modulus.json": sha256(out_dir / "outer_boundary_modulus.json"),
        },
        "env": {k: os.environ.get(k, "") for k in ["PYTHONPATH"]},
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()


