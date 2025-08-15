#!/usr/bin/env python3
from __future__ import annotations

import cmath
import numpy as np

from rdm.colligation import prime_grid_lossless, kyp_lossless_equalities, transfer_H, scalar_port
from rdm.pick import pick_kernel, is_psd
from rdm.rsm import FiniteLedgerBlock, recognition_spectral_measure, laplace_admittance, prime_window_block
from rdm.outer import poisson_outer_from_modulus
from rdm.doors import certify_pick_on_interval
from rdm.det2 import boundary_modulus_log, det2_from_primes, xi_completed, first_primes
import json
import pathlib
import hashlib
import os
import sys
import time
import random
import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RDM-IS driver: artifacts and PSD certification")
    p.add_argument("--primes", type=int, default=1000, help="number of initial primes")
    p.add_argument("--intervals", type=str, default="0,10;10,20;20,30", help="semicolon-separated T1,T2 pairs")
    p.add_argument("--eps", type=float, default=1e-2, help="epsilon for boundary line (outer)")
    p.add_argument("--sigmas", type=str, default="5e-3,2e-3,1e-3", help="comma-separated sigma schedule for D6")
    p.add_argument("--nodes", type=str, default="96,128,192", help="comma-separated grid sizes for D6")
    p.add_argument("--use-outer", action="store_true", help="use outer normalization in Θ-from-J certification")
    p.add_argument("--dps", type=int, default=150, help="mpmath precision (decimal places)")
    p.add_argument("--outdir", type=str, default="artifacts", help="base output directory")
    p.add_argument("--outer-samples", type=int, default=2048, help="samples for Poisson outer integral")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Deterministic seed for reproducibility
    seed = 123456789
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    timings = {}

    t0 = time.perf_counter()
    # Toy prime grid and unit port (C1, C2)
    # Scale primes for RH tests (pilot size; increase as needed)
    primes = list(first_primes(int(args.primes)))
    A, B, C, D, P = prime_grid_lossless(primes)
    assert kyp_lossless_equalities(A, B, C, D, P), "Lossless KYP equalities failed"
    timings["C1_C2_lossless_kyp"] = time.perf_counter() - t0

    # Build scalar Schur θ(s) = H(s) (lossless) at a few nodes near the boundary
    def theta(s: complex) -> complex:
        H = transfer_H(A, B, C, D, s)
        # single-port extraction (first basis vector)
        e1 = np.zeros((len(primes), 1), dtype=complex)
        e1[0, 0] = 1.0
        h = scalar_port(H, e1, e1)
        # Use h directly as Schur (|h|<=1)
        return h

    t1 = time.perf_counter()
    sigma = 1e-2
    t_grid = np.linspace(0.0, 10.0, 64)
    s_vals = 0.5 + sigma + 1j * t_grid
    # Shift to z-domain (Re z>0) for the lossless transfer; evaluate θ(s)=θ_z(s-1/2)
    theta_vals = np.array([theta(complex(s - 0.5)) for s in s_vals])
    K = pick_kernel(theta_vals, s_vals)
    ok = is_psd(K, tol=1e-10)
    print(f"Pick PSD on toy grid: {ok}; min eig={np.linalg.eigvalsh(K).min():.3e}")
    print(f"[timing] D6_pick_toy: {timings['D6_pick_toy']:.3f}s")
    timings["D6_pick_toy"] = time.perf_counter() - t1

    # Artifact saving to per-run stamped directory (E2)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    base_out = pathlib.Path(args.outdir)
    out_dir = base_out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "grid_s_vals.npy", s_vals)
    np.save(out_dir / "grid_theta_vals.npy", theta_vals)
    np.save(out_dir / "grid_pick_K.npy", K)

    # μ_RS toy test + outer normalization demo (artifacts preview)
    block = FiniteLedgerBlock(tick_fluxes=((0, 1.0), (1, 1.0)))
    mu = recognition_spectral_measure(block)
    t2 = time.perf_counter()

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
    print(f"[timing] B3_outer_demo: {timings['B3_outer_demo']:.3f}s")
    timings["B3_outer_demo"] = time.perf_counter() - t2

    # D6.PICK certification on multiple intervals (export log)
    def parse_intervals(spec: str):
        out = []
        for part in spec.split(";"):
            if not part.strip():
                continue
            a, b = part.split(",")
            out.append((float(a), float(b)))
        return out

    intervals = parse_intervals(args.intervals)
    certs = []
    all_ok = True
    t3 = time.perf_counter()
    for (T1, T2) in intervals:
        cert = certify_pick_on_interval(theta, T1=T1, T2=T2)
        cert["interval"] = [float(T1), float(T2)]
        certs.append(cert)
        all_ok = all_ok and bool(cert["ok"])
    with open(out_dir / "pick_certification.json", "w") as f:
        json.dump({"ok": all_ok, "certs": certs}, f, indent=2)
    print(f"Pick certification overall ok={all_ok}")
    print(f"[timing] D6_pick_lossless_intervals: {timings['D6_pick_lossless_intervals']:.3f}s")
    timings["D6_pick_lossless_intervals"] = time.perf_counter() - t3

    # B2.PORT μ_RS from a prime-window block (integrated)
    t4 = time.perf_counter()
    pw_block = prime_window_block(tuple(primes), weight="log")
    mu_pw = recognition_spectral_measure(pw_block)
    s = 0.5 + 0.05 + 0j
    phi_pw = laplace_admittance(mu_pw, s)
    with open(out_dir / "mu_rs_prime_window.json", "w") as f:
        json.dump({"s": [s.real, s.imag], "phi_pw": [phi_pw.real, phi_pw.imag], "support": getattr(mu_pw, "support")}, f, indent=2)
    timings["B2_mu_rs_prime_window"] = time.perf_counter() - t4
    print(f"[timing] B2_mu_rs_prime_window: {timings['B2_mu_rs_prime_window']:.3f}s")

    # B3.OUTER (boundary modulus) from det2/xi along Re s = 1/2+ε
    t5 = time.perf_counter()
    eps = float(args.eps)
    t_min = min(T1 for (T1, _T2) in intervals)
    t_max = max(T2 for (_T1, T2) in intervals)
    t_grid_outer = np.linspace(t_min, t_max, max(400, args.outer_samples // 4))
    u_vals = [boundary_modulus_log(0.5, eps, float(t), tuple(primes)) for t in t_grid_outer]
    with open(out_dir / "outer_boundary_modulus.json", "w") as f:
        json.dump({"sigma0": 0.5, "eps": eps, "t": t_grid_outer.tolist(), "u": u_vals}, f, indent=2)
    timings["B3_outer_det2_xi"] = time.perf_counter() - t5
    print(f"[timing] B3_outer_det2_xi: {timings['B3_outer_det2_xi']:.3f}s")

    # Optional: build outer from the boundary modulus and use Jhat := det2/(O*xi)
    O_from_det2_xi = None
    if args.use_outer:
        def u_eps_lambda(t: float) -> float:
            return float(boundary_modulus_log(0.5, eps, float(t), tuple(primes)))

        O_from_det2_xi = poisson_outer_from_modulus(
            u=u_eps_lambda, sigma0=0.5, t_support=(t_min, t_max), samples=int(args.outer_samples)
        )
        # Save a few outer samples along a vertical line
        s_probe = 0.5 + eps + 1j * np.linspace(t_min, t_max, 50)
        O_vals = [complex(O_from_det2_xi(complex(s))) for s in s_probe]
        with open(out_dir / "outer_O_samples.json", "w") as f:
            json.dump({
                "s": [[float(s.real), float(s.imag)] for s in s_probe],
                "O": [[v.real, v.imag] for v in O_vals]
            }, f, indent=2)

    # RH boundary PSD with Θ-from-J or Θ-from-Jhat (det2/ξ with optional outer)
    def theta_from_J(s: complex, primes_tuple: tuple[int, ...]) -> complex:
        try:
            import mpmath as mp
            mp.dps = int(args.dps)
        except Exception:
            pass
        denom = xi_completed(s, primes_tuple) + 1e-30
        if args.use_outer and O_from_det2_xi is not None:
            denom = (O_from_det2_xi(s) * denom)
        J = det2_from_primes(s, primes_tuple) / denom
        z = (2.0 * J - 1.0) / (2.0 * J + 1.0)
        return z

    # Wrapper to accommodate doors.certify_pick_on_interval shift
    theta_J_wrapped = lambda z: theta_from_J(z + 0.5, tuple(primes))
    t6 = time.perf_counter()
    rh_certs = []
    rh_ok = True
    # Parse schedules from args; ensure sigmas >= eps if using outer data at eps
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    nodes = [int(x) for x in args.nodes.split(",") if x.strip()]
    if args.use_outer:
        sigmas = [max(s, eps + 1e-6) for s in sigmas]
    for (T1, T2) in intervals:
        cert = certify_pick_on_interval(
            theta_J_wrapped, T1=T1, T2=T2, sigmas=tuple(sigmas), ns=tuple(nodes), tol=1e-9
        )
        cert["interval"] = [float(T1), float(T2)]
        rh_certs.append(cert)
        rh_ok = rh_ok and bool(cert["ok"])
    with open(out_dir / "rh_pick_certification.json", "w") as f:
        json.dump({"ok": rh_ok, "certs": rh_certs}, f, indent=2)
    print(f"RH boundary PSD (Θ-from-J) overall ok={rh_ok}")
    timings["D6_pick_theta_from_J"] = time.perf_counter() - t6
    print(f"[timing] D6_pick_theta_from_J: {timings['D6_pick_theta_from_J']:.3f}s")

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
        "timings_sec": timings,
        "artifacts": {
            "grid_s_vals.npy": sha256(out_dir / "grid_s_vals.npy"),
            "grid_theta_vals.npy": sha256(out_dir / "grid_theta_vals.npy"),
            "grid_pick_K.npy": sha256(out_dir / "grid_pick_K.npy"),
            "outer_demo.json": sha256(out_dir / "outer_demo.json"),
            "pick_certification.json": sha256(out_dir / "pick_certification.json"),
            "mu_rs_prime_window.json": sha256(out_dir / "mu_rs_prime_window.json"),
            "outer_boundary_modulus.json": sha256(out_dir / "outer_boundary_modulus.json"),
            "rh_pick_certification.json": sha256(out_dir / "rh_pick_certification.json"),
        },
        "env": {k: os.environ.get(k, "") for k in ["PYTHONPATH"]},
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()


