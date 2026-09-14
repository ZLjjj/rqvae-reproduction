<callout emoji="white_check_mark" background-color="light-green" border-color="green">
**实验结论**

- **优先候选：任务3 L1 Sinkhorn**。L1 利用率由 7.71% 提升到 100%，L1 最大频次由 6,962 降到 216；L2-L4 利用率仍保持 100%，完整 SID ICR 达到 0.999093。
- **对照候选：任务2 L1 Sinkhorn**。完整 SID ICR 略高，为 0.999254，但 L2/L3 利用率下降到 90.72%/79.88%。
- **不采用实验 B/C**：单纯缩小 L1 codebook 没有解决利用率问题，并使完整 SID 指标退化。
- **暂不采用实验 D**：L1 虽达到 100% 利用率，但容量压力转移到 L3/L4，出现后层集中。
- 正式替换前，任务3 L1 Sinkhorn 仍需独立随机种子复现，并补充全量内容一致性指标。
</callout>
