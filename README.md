# GenSearchRec

基于多种量化策略（RVQ/FSQ/RQ-OPQ）的自编码模型，对高维嵌入进行离散压缩与重建，适用于搜索/推荐场景的嵌入压缩、离线编码与重建评估。

## 特性
- **多量化策略**：RQVAE（残差 VQ）、FSQVAE（有限标量量化）、RQ-OPQ-VAE（残差 OPQ）。
- **可选增强**：KMeans 初始化、Sinkhorn 软分配、旋转矩阵优化。
- **灵活架构**：可配置 MLP 编解码器（Dropout/BN/激活）。
- **端到端脚本**：覆盖数据准备、训练、索引生成和评估。

## 目录与关键模块
- **模型库**：
  - [sid/models/rqvae.py](sid/models/rqvae.py): RQVAE 主干
  - [sid/models/fsq_vae.py](sid/models/fsq_vae.py): FSQVAE 主干
  - [sid/models/rqopq_vae.py](sid/models/rqopq_vae.py): RQ-OPQ-VAE 主干
  - [sid/models/vq.py](sid/models/vq.py), [sid/models/rq.py](sid/models/rq.py): 基础 VQ 与残差 VQ
  - [sid/models/fsq.py](sid/models/fsq.py): FSQ 核心
  - [sid/models/rq_opq.py](sid/models/rq_opq.py): OPQ 核心
  - [sid/models/layers.py](sid/models/layers.py): MLP、KMeans、Sinkhorn
  - [sid/models/datasets.py](sid/models/datasets.py): `EmbDataset` 数据加载与清洗
- **训练脚本**：
  - [sid/rqvae_run.py](sid/rqvae_run.py) / [sid/rqvae_run.sh](sid/rqvae_run.sh): 训练 RQVAE
  - [sid/fsq_run.py](sid/fsq_run.py): 训练 FSQVAE
  - [sid/rqopq_run.py](sid/rqopq_run.py) / [sid/rqopq_run.sh](sid/rqopq_run.sh): 训练 RQ-OPQ
  - [sid/rqkmeans_plus_run.py](sid/rqkmeans_plus_run.py) / [sid/rqkmeans_plus_run.sh](sid/rqkmeans_plus_run.sh): KMeans++ 变体
  - [sid/trainer.py](sid/trainer.py): 统一训练/评估/保存
  - [sid/utils.py](sid/utils.py): 通用工具
- **数据与索引脚本**：
  - [sid/datasets/meta2emb.py](sid/datasets/meta2emb.py): 原始文本/JSON 转嵌入
  - [sid/datasets/generate_indices.py](sid/datasets/generate_indices.py): 用训练模型生成离散索引
  - [sid/datasets/generate_indices_plus.py](sid/datasets/generate_indices_plus.py): 变体索引生成
  - [sid/datasets/generate_indices_opq.py](sid/datasets/generate_indices_opq.py): RQ-OPQ 索引生成
  - Shell 示例：[sid/datasets/run_meta2emb.sh](sid/datasets/run_meta2emb.sh)，[sid/datasets/run_generate_indices.sh](sid/datasets/run_generate_indices.sh)，[sid/datasets/run_generate_indices_plus.sh](sid/datasets/run_generate_indices_plus.sh)，[sid/datasets/run_generate_indices_opq.sh](sid/datasets/run_generate_indices_opq.sh)

## 环境准备
- 建议 Python 3.9+
- 主要依赖：torch、numpy、scikit-learn、tqdm、transformers

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scikit-learn tqdm transformers
```

## 使用流程

### 1) 数据准备（嵌入生成）
若只有原始文本/JSON，请先生成 `.npy` 嵌入矩阵。

最简方式：修改并运行 [sid/datasets/run_meta2emb.sh](sid/datasets/run_meta2emb.sh)。

或直接调用 Python：
```bash
python sid/datasets/meta2emb.py \
  --inputs data/raw.jsonl \
  --output_prefix data/my_embeddings \
  --text_field meta \
  --max_workers 64
```
输出：`data/my_embeddings.npy` 与 `data/my_embeddings_meta.jsonl`（行号一一对应）。

参数说明：
| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--inputs` | 输入文件路径（支持多文件、目录、逗号分隔） | **必填** |
| `--output_prefix` | 输出前缀，自动生成 `.npy` 与 `_meta.jsonl` | **必填** |
| `--text_field` | JSON 中包含文本的字段名 | `meta` |
| `--model_name` | API 模型名称 | `base_model` |
| `--max_workers` | 并发线程数 | `128` |
| `--api_url` | Embedding API 服务地址 | (脚本内置默认值) |

### 2) 模型训练
在运行前修改脚本内数据路径、输出目录、训练参数。

- RQVAE：`bash sid/rqvae_run.sh`（或 `python sid/rqvae_run.py --data_path ...`）
- RQ-OPQ：`bash sid/rqopq_run.sh`
- FSQ：`python sid/fsq_run.py --data_path ...`
- KMeans++ 变体：`bash sid/rqkmeans_plus_run.sh`

训练器会在 `ckpt_dir` 下保存最佳损失与最佳碰撞率的权重。

### 3) 索引生成
使用训练好的权重为全量数据生成离散 Code ID。

- 通用：`bash sid/datasets/run_generate_indices.sh`
- Plus 变体：`bash sid/datasets/run_generate_indices_plus.sh`
- RQ-OPQ：`bash sid/datasets/run_generate_indices_opq.sh`

请在脚本内设置 `DATASET`、`CKPT_PATH`、`DEVICE` 等参数。

## Python 最小示例
```python
import torch
from torch.utils.data import DataLoader
from sid.models.datasets import EmbDataset
from sid.models.rqvae import RQVAE
from sid.trainer import Trainer

ds = EmbDataset("data/my_embeddings.npy")
loader = DataLoader(ds, batch_size=256, shuffle=True)

model = RQVAE(
    in_dim=ds.dim,
    num_emb_list=[256, 256, 256],
    e_dim=64,
    layers=[512, 256]
)

args = type("Args", (), {
    "lr": 1e-3,
    "epochs": 10,
    "learner": "adamw",
    "lr_scheduler_type": "linear",
    "weight_decay": 1e-4,
    "warmup_epochs": 1,
    "save_limit": 5,
    "eval_step": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt_dir": "checkpoints",
})()

trainer = Trainer(args=args, model=model, data_num=len(loader))
trainer.fit(loader)

batch = next(iter(loader))
indices = model.get_indices(batch, use_sk=False)  # [batch, num_quantizers]
```

## 评估指标
- **碰撞率 (collision rate)**：`(N - unique_codes) / N`，越低越好（见 [sid/trainer.py](sid/trainer.py)）。
- **重构损失**：MSE 重构误差。

## 常见问题
- 路径：运行脚本前确认数据与 checkpoint 路径已在脚本内更新。
- 维度：`in_dim` 应与数据的最后一维一致。
- 初始化：残差类模型可开启 `kmeans_init=True` 获得更平稳的收敛。
- Sinkhorn：`sk_epsilons=0` 表示硬分配，温度越小越接近硬分配但可能不稳定。

