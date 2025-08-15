# Methods: RDM-IS (Recognition-Led Deductive Measurement Instruction Set)

This document codifies the 16-opcode ISA and the frozen normalizations used in the RDM pipeline.

## Frozen normalizations
- Cayley: Θ = (H − 1) / (H + 1), H = (1 + Θ)/(1 − Θ)
- Half-plane Pick kernel: K(s,w) = (1 − Θ(s) Θ(w)*) / (s + w* − 1)
- Domain: right half-plane Re s > 1/2 for RH; use shift z = s − 1/2 for lossless state-space where needed

## Opcodes
- A1.SEED, A2.LEDGER, A3.CADENCE, A4.SIMILARITY
- B1.DICT, B2.PORT (μ_RS), B3.OUTER
- C1.FOT (finite lossless colligation), C2.KYP (passivity certificates)
- D1.COMPACT, D2.DESMOOTH, D3.CARLESON / D4.ARCTAN / D5.DESM-POS, D6.PICK
- E1.LIFT, E2.AUDIT, E3.LOOP

## Proof-carrying outputs
- Finite-stage KYP logs, Θ samples, Pick matrices and eigenvalues, outer modulus snapshots, run manifest (hashes, seeds, versions).

## Analytic bridge (paper corollary)
- If Θ is Schur on Ω:=\{Re s>1/2\}, then the half‑plane Pick kernel K_Θ(s,w):=(1−Θ(s)Θ(w)*)/(s+w*−1) is PSD for any finite node set (half‑plane Nevanlinna–Pick equivalence).
- See analytic track (paper) corollary “Θ Schur ⇒ Pick‑PSD on Ω.” RDM’s Pick matrices are finite‑grid instances/witnesses of this fact.

## Minimal CLI flow for Pick matrices and manifest
- Run the driver (creates a stamped artifact dir `artifacts/YYYYMMDD-HHMMSS/`):
  - PYTHONPATH="." python3 scripts/run_rdm_is.py --use-outer --primes 200 --intervals "0,10"
- Artifacts saved per run include:
  - grid_s_vals.npy, grid_theta_vals.npy, grid_pick_K.npy
  - pick_certification.json (per‑interval PSD results)
  - manifest.json (hashes, Python version, seed, intervals, timings)

## Notes on interpretation
- When Θ in the driver is sourced from the analytic object, PSD of K_Θ is guaranteed by the corollary; artifacts serve as reproducible witnesses.
- When Θ is built numerically (det₂/ξ with provisional outer), artifacts are diagnostics; PSD failures typically indicate discretization/normalization issues, not a contradiction to the analytic result.
