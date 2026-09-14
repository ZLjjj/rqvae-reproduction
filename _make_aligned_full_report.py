from pathlib import Path
import re

root = Path(r"C:\Users\dszlj\Desktop\gensearchrec-main-0901")
src = (root / "sid_three_way_feishu.md").read_text(encoding="utf-8")

replacement = '''## 4. 核心结果

| 基础Embedding模型 | SID生成方法 | 关键参数设置 | 梯度更新方式 | 冲突率 | 独立编码率 | 总体内聚性 | 视频内聚性 | 电台内聚性 |
|---|---|---|---|---:|---:|---|---|---|
| Qwen3-Embedding-4B | RQ-VAE + 硬编码（baseline 视频子集） | 码本：4层1024维；硬编码层：1层；混合 station + video 训练后取 video | 指数移动平均更新 EMA（decay=0.99） | 0.038667 | 0.961333 | 共享一层：0.575612；共享两层：0.667208；共享三层：0.755617 | 共享一层：0.575612；共享两层：0.667208；共享三层：0.755617 | 不适用 |
| Qwen3-Embedding-4B | RQ-VAE + 硬编码（任务2 同源 video-only） | 码本：4层1024维；硬编码层：1层；e_dim=128 | 指数移动平均更新 EMA（decay=0.99） | 0.004875 | 0.995125 | 共享一层：0.547222；共享两层：0.695476；共享三层：0.839722 | 共享一层：0.547222；共享两层：0.695476；共享三层：0.839722 | 不适用 |
| Qwen3-Embedding-4B cleaned text_format | RQ-VAE + 硬编码（任务3 cleaned video-only） | 码本：4层1024维；硬编码层：1层；e_dim=128；清洗重组 text_format | 指数移动平均更新 EMA（decay=0.99） | 0.004890 | 0.995110 | 共享一层：0.620066；共享两层：0.755967；共享三层：0.850521 | 共享一层：0.620066；共享两层：0.755967；共享三层：0.850521 | 不适用 |

<callout emoji="✅" background-color="light-green" border-color="green">
**video-only 的收益**：任务2相较 baseline 的 CR 从 3.8667% 降到 0.4875%，下降约 87.4%；这说明训练域从混合 station + video 收敛到 video-only 是最主要的收益来源。
</callout>
'''

pattern = r"## 4\. 核心结果\n.*?(?=## 5\. Prefix bucket 分布)"
out = re.sub(pattern, replacement + "\n", src, count=1, flags=re.S)
(root / "sid_three_way_feishu_full_aligned.md").write_text(out, encoding="utf-8")
print(root / "sid_three_way_feishu_full_aligned.md")
