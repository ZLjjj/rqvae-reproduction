# SID实验中的硬编码层(Hardcode)分析

## 核心问题：哪些实验的最后一层有硬编码？

通过检查各实验的 `indices.jsonl` 文件，可以明确看出不同实验在最后一层（L5/`<e_x>`）的处理方式：

---

## 有硬编码层的实验

### 1. RQ-KMeans 系列 ✅ 有硬编码

**实验文件**：
- `results_three_way/rqkmeans_task2/indices.jsonl`
- `results_three_way/rqkmeans_task3/indices.jsonl`

**特征**：
- 每条记录包含 `"saletype"` 字段
- L5 的取值范围有限（通常 42-50 等少数值）
- 硬编码内容：`saletype`（付费类型）

**示例记录**：
```json
{
  "indices": [853, 475, 672, 353, 42],
  "tokens": ["<a_853>", "<b_475>", "<c_672>", "<d_353>", "<e_42>"],
  "saletype": "2"
}
```

**硬编码映射逻辑**：
- `saletype = "1"` (FREE/免费) → L5 index ≈ 49
- `saletype = "2"` (PAY/付费) → L5 index ≈ 42, 50 等
- L5 直接由业务字段 `pay_type` 决定，不通过向量学习

**码本配置**：
- L1-L4: [1024, 1024, 1024, 1024] - 向量量化
- L5: 512 - 其中部分码位用于硬编码 saletype

---

### 2. 任务2/任务3 + L1 Sinkhorn 系列 ✅ 有硬编码

**实验文件**：
- `results_three_way/l1_screen_task2_eps003/indices/indices.jsonl`
- `results_three_way/exp_task3_l1_sinkhorn/indices/indices.jsonl`

**特征**：
- 每条记录包含 `"saletype"` 字段
- L5 的取值范围有限
- 硬编码内容：`saletype`（付费类型）

**示例记录**：
```json
{
  "indices": [731, 403, 487, 460, 42],
  "tokens": ["<a_731>", "<b_403>", "<c_487>", "<d_460>", "<e_42>"],
  "saletype": "2"
}
```

**码本配置**：
- L1-L4: [1024, 1024, 1024, 1024] - 向量量化 + Sinkhorn
- L5: 256 - 使用 `HardCodeMapper` 编码 saletype

---

### 3. 任务2/任务3 原始方案 ✅ 有硬编码

**实验文件**：
- `results_three_way/same_source_video/indices/indices.jsonl`（任务2原始）
- `results_three_way/video_v3/indices/indices.jsonl`（任务3原始）

**特征**：
- 每条记录包含 `"saletype"` 字段
- 硬编码内容：`saletype`

**码本配置**：
- L1-L4: [1024, 1024, 1024, 1024]
- L5: 256 - 硬编码

---

## 无硬编码层的实验

### 1. RQ-OPQ 系列 ❌ 无硬编码

**实验文件**：
- `results_three_way/rqopq_task2/indices.jsonl`
- `results_three_way/rqopq_task3/indices.jsonl`

**特征**：
- 记录**不包含** `"saletype"` 字段
- L5 的取值范围广（0-511 均可能）
- L5 完全由 OPQ 优化后的向量量化决定

**示例记录**：
```json
{
  "indices": [853, 475, 672, 747, 824],
  "tokens": ["<a_853>", "<b_475>", "<c_672>", "<d_747>", "<e_824>"],
  "sid": ["<a_853>", "<b_475>", "<c_672>", "<d_747>", "<e_824>"]
}
```

**码本配置**：
- L1-L4: [1024, 1024, 1024, 1024] - FAISS RQ
- L5: 512 - OPQ 旋转优化后的向量量化（**纯向量学习，无业务字段**）

---

## 对比总结表

| 实验方案 | L5是否硬编码 | 硬编码内容 | L5码本大小 | 有saletype字段 |
|---|:---:|---|:---:|:---:|
| **任务3+L1 Sinkhorn** | ✅ 是 | saletype | 256 | ✅ |
| **任务2+L1 Sinkhorn** | ✅ 是 | saletype | 256 | ✅ |
| **RQ-OPQ 任务3** | ❌ 否 | 无 | 512 | ❌ |
| **RQ-OPQ 任务2** | ❌ 否 | 无 | 512 | ❌ |
| **RQ-KMeans 任务3** | ✅ 是 | saletype | 512 (部分) | ✅ |
| **RQ-KMeans 任务2** | ✅ 是 | saletype | 512 (部分) | ✅ |
| 任务3原始 | ✅ 是 | saletype | 256 | ✅ |
| 任务2原始 | ✅ 是 | saletype | 256 | ✅ |
| Baseline混合域 | ✅ 是 | saletype + publish_year | 256 | ✅ |

---

## 硬编码的优缺点分析

### ✅ 硬编码的优点

1. **强制语义分组**
   - 免费内容和付费内容被严格分开
   - 便于构建付费状态相关的检索策略

2. **节省码本容量**
   - saletype 只有 2-3 个取值，不需要占用完整的 256/512 码位
   - 可以用更多容量表示语义信息

3. **解释性强**
   - L5 的某些码位直接对应业务属性
   - 便于调试和理解 SID 结构

### ❌ 硬编码的缺点

1. **灵活性降低**
   - 如果业务字段变化（如增加新的付费类型），需要重新设计编码
   - 无法自动学习字段之间的潜在关联

2. **跨域泛化能力弱**
   - 硬编码字段与特定业务强绑定
   - 迁移到其他领域时需要修改 L5 逻辑

3. **容量利用率问题**
   - RQ-KMeans 的 L5 利用率只有 47.8%（见实验表格）
   - 部分码位被硬编码占用后无法用于语义表达

---

## 为什么 RQ-OPQ 不用硬编码？

RQ-OPQ 采用 **FAISS 原生 RQ-OPQ** 实现，其最后一层通过 **正交旋转优化（Orthogonal Product Quantization）** 自动学习最优量化方式。

**核心思想**：
- L1-L4 使用标准 Residual Quantization
- L5 在残差向量上应用 OPQ 旋转矩阵
- 旋转后的向量分布更均匀，更适合 K-Means 量化

**结果**：
- L5 的 512 个码位**全部用于语义表达**
- 没有预留位置给 saletype
- 依靠前 4 层的语义聚类隐式区分付费状态

---

## 实际使用建议

### 场景1：需要强业务约束
**推荐**：任务3+L1 Sinkhorn（有硬编码）
- 适用于需要严格区分付费/免费的推荐场景
- L5 可以直接用于付费状态过滤

### 场景2：追求最大语义表达能力
**推荐**：RQ-OPQ（无硬编码）
- L5 的 512 个码位全部用于细粒度语义
- 训练速度快（3-5分钟）
- ICR=99.22% 已接近 99.91%

### 场景3：平衡训练速度和精度
**推荐**：RQ-KMeans（有硬编码）
- 训练速度快（3-5分钟）
- 保留 saletype 硬编码
- ICR=98.36% 仍然很高

---

## 附录：硬编码实现代码位置

### HardCodeMapper 类
**文件**：`sid/src/models/rqvae.py`

```python
class HardCodeMapper:
    def __init__(self, num_emb=256):
        self.num_emb = num_emb
    
    def encode(self, saletype, publish_year=None):
        # 将 saletype (1=FREE, 2=PAY) 映射到 [0, num_emb-1]
        # 可选：结合 publish_year
        pass
    
    def decode(self, index):
        # 从 index 反推 saletype
        pass
```

### 调用位置
**训练脚本**：`sid/scripts/train/train_rqvae.py`
```python
model = RQVAE(
    ...,
    num_emb_list=[1024, 1024, 1024, 1024, 256],
    use_hardcode_layer=True,  # 启用硬编码
    hardcode_fields=['saletype']
)
```

**索引生成**：`sid/src/codebook/generate_indices_rqvae.py`
```python
# 从 metadata 中读取 pay_type
saletype = "2" if item["pay_type"] == "PAY" else "1"
# 添加到输出
result["saletype"] = saletype
```
