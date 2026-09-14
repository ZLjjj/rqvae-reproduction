# Successful Runs

## 2026-09-10: Task3 L1 Sinkhorn epsilon 0.003 screening

- Goal: reproduce the promising task2 candidate on cleaned text_format embeddings.
- Data: `3_text_format_embedding.npy` and `3_meta.jsonl`, 193,034 rows.
- Environment: CPU, PyTorch 2.14.0+cpu, seed 2024.
- Settings: `num_emb_list=[1024,1024,1024,1024,256]`, `sk_epsilons=[0.003,0,0,0.003,0]`, EMA decay 0.99, 10 screening epochs.
- Baseline: task3 original `[0,0,0,0.003,0]`.
- Result: L1 utilization `7.71% -> 100%`, L1 max frequency `6962 -> 216`; L2-L4 utilization remained `100%`; full ICR `0.995110 -> 0.999093`; full CR `0.004890 -> 0.000907`.
- Content sample: candidate L1 bucket maximum is 216; global content purity requires a separate aggregate run.
- Reproduction status: reproduced once
- Related files: `results_three_way/exp_task3_l1_sinkhorn/checkpoints/best_collision_model.pth`, `results_three_way/exp_task3_l1_sinkhorn/indices/indices.jsonl`, `results_three_way/exp_task3_l1_sinkhorn/comparison.json`, `results_three_way/exp_task3_l1_sinkhorn/content_analysis.md`
- Conclusion: successful screening result; candidate is stronger than the task2 L1 Sinkhorn run because L2-L4 utilization did not drop.
