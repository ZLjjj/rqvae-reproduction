# Decisions

## 2026-09-09: L1 utilization improvement screening

- Keep the existing RQVAE structure and codebook sizes.
- Compare the current Sinkhorn schedule `[0, 0, 0, 0.003, 0]` against `[0.001, 0, 0, 0.003, 0]`.
- Only L1 Sinkhorn is changed; L4 remains unchanged.
- Accept a change only if L1 active-code coverage improves without increasing full-SID CR or materially reducing L1 content consistency.
