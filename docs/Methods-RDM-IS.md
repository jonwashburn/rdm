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
