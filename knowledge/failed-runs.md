# Failed Runs

## 2026-09-09: L1 Sinkhorn epsilon 0.001 screening

- Goal: increase L1 codebook coverage with minimal model changes.
- Data version: `2_meta_full_video.npy`
- Environment: CPU, PyTorch 2.14.0+cpu
- Command: `train_rqvae.py ... --sk_epsilons 0.001 0 0 0.003 0 --epochs 10`
- Baseline: `[0, 0, 0, 0.003, 0]`
- Result: failed during the first epoch with `Sinkhorn Algorithm returns nan/inf values`.
- Why it failed: epsilon was too small for the centered distance scale and caused numerical underflow/overflow in `exp(-distance / epsilon)`.
- Reproduction status: reproduced once
- Related files: `sid/src/models/vq_mix.py`, `knowledge/decisions.md`
- Conclusion: do not use L1 epsilon=0.001 without numerical stabilization; screen epsilon=0.003 next.

## 2026-09-09: L1 Sinkhorn epsilon 0.003 screening (paused)

- Goal: increase L1 codebook coverage with minimal model changes.
- Data version: `2_meta_full_video.npy`
- Environment: CPU, PyTorch 2.14.0+cpu
- Command: `train_rqvae.py ... --sk_epsilons 0.003 0 0 0.003 0 --epochs 10`
- Baseline: `[0, 0, 0, 0.003, 0]`
- Result: paused by user after epoch 8/10; best observed collision rate was about 0.002865 at epoch 6.
- Why it is incomplete: candidate SID generation and L1 content analysis have not been run yet.
- Reproduction status: unverified
- Related files: `results_three_way/l1_screen_task2_eps003/checkpoints/`
- Conclusion: keep the checkpoint and resume evaluation tomorrow; do not promote yet.

## 2026-09-10: L1 Sinkhorn epsilon 0.003 screening evaluation

- Goal: increase L1 codebook coverage while preserving or improving full SID uniqueness.
- Data version: `2_meta_full_video.npy` and `2_meta_full_video.jsonl`
- Environment: CPU, PyTorch 2.14.0+cpu
- Code revision: incremental `sk_epsilons` configuration only; no model structure change.
- Command: `train_rqvae.py ... --epochs 10 --sk_epsilons 0.003 0 0 0.003 0`; run was paused after epoch 8.
- Baseline: task2 checkpoint with `[0, 0, 0, 0.003, 0]`.
- Candidate checkpoint: `results_three_way/l1_screen_task2_eps003/checkpoints/best_collision_model.pth`
- Candidate SID: `results_three_way/l1_screen_task2_eps003/indices/indices.jsonl`
- Result: full SID CR `0.000746`, ICR `0.999254`, versus baseline CR `0.004875`, ICR `0.995125`.
- L1 utilization: `4.79% -> 100.00%`; L1 active codes `49 -> 1024`; L1 max code frequency `9511 -> 210`.
- Other layer utilization: L2 `100% -> 90.72%`, L3 `100% -> 79.88%`, L4 remains `100%`; L5 unchanged `45.51%`.
- L1 bucket content examples: candidate top buckets have roughly 200 items each; purity varies by bucket, e.g. 90.48%/90.48%/71.43% for video type/genre/pay status in one mixed sports/fitness bucket and 99.51%/99.51%/96.59% in a music bucket.
- Why incomplete: only 8 of planned 10 screening epochs completed; task3 has not yet received the same candidate configuration; content analysis is illustrative top-bucket sampling, not a global aggregate.
- Reproduction status: unverified
- Related files: `results_three_way/l1_screen_task2_eps003/comparison.json`, `results_three_way/l1_screen_task2_eps003/content_analysis.md`
- Conclusion: promising for full SID uniqueness and L1 load balancing, but must be evaluated against L2/L3 utilization and global content purity before promotion.

## 2026-09-10: Experiment B L1 codebook 256

- Goal: reduce unused L1 capacity without changing assignment method.
- Data: `2_meta_full_video.npy`; 10 epochs CPU screening; seed 2024.
- Settings: `num_emb_list=[256,1024,1024,1024,256]`, `sk_epsilons=[0,0,0,0.003,0]`; EMA and KMeans unchanged.
- Result: L1 active codes=6 (2.34% of 256), L1 max frequency=45,311; full CR=0.006999, ICR=0.993001.
- Baseline: task2 full model CR=0.004875, ICR=0.995125, L1 active codes=49/1024.
- Conclusion: shrinking L1 alone worsened collapse and full SID uniqueness; do not promote.

## 2026-09-10: Experiment C L1 codebook 128

- Goal: test a smaller L1 capacity.
- Data/settings: same as Experiment B, `num_emb_list=[128,1024,1024,1024,256]`.
- Result: L1 active codes=14 (10.94% of 128), L1 max frequency=19,169; full CR=0.005548, ICR=0.994452.
- Conclusion: better than Experiment B but still worse than the original task2 model; do not promote.

## 2026-09-10: Experiment D L1 codebook 256 plus Sinkhorn

- Goal: combine reduced L1 capacity with L1 balancing.
- Data/settings: same data, `num_emb_list=[256,1024,1024,1024,256]`, `sk_epsilons=[0.003,0,0,0.003,0]`, 10 epochs CPU.
- Result: L1 active codes=256 (100%), L1 max frequency=785; full CR=0.004056, ICR=0.995944. However L3/L4 active codes fell to 393/399 and L4 max frequency rose to 47,502.
- Conclusion: L1 became balanced, but residual capacity collapsed into later-layer hotspots; not suitable without further stabilization.
