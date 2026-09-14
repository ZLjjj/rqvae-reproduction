# HardCode 硬编码层构建详解

本文档详细解释GenSearchRec项目中硬编码层(HardCodeMapper)的构建原理、实现细节和使用方式。

---

## 📋 目录

1. [什么是HardCode](#什么是hardcode)
2. [为什么需要HardCode](#为什么需要hardcode)
3. [HardCodeMapper实现](#hardcodemapper实现)
4. [编码规则详解](#编码规则详解)
5. [在RQVAE中的使用](#在rqvae中的使用)
6. [训练和推理流程](#训练和推理流程)
7. [实际效果分析](#实际效果分析)

---

## 什么是HardCode

### 基本概念

**HardCode（硬编码）** 是指在RQVAE的第5层（L5）不使用向量量化学习，而是直接根据业务字段（如付费类型、出版年份）计算出固定的索引值。

```
传统VQ第5层:
  残差向量(64维) → 距离计算 → 选择最近码本 → 索引idx_5

HardCode第5层:
  业务字段(saletype, publish_year) → 规则映射 → 索引idx_5
```

### 关键特点

1. **不参与梯度训练**: L5的码本不通过反向传播学习
2. **确定性映射**: 相同的业务字段总是映射到相同的索引
3. **压缩业务信息**: 将多个字段打包到一个索引中
4. **向量为零**: L5的量化向量在实现中设为0（不影响重构）

---

## 为什么需要HardCode

### 动机1: 强制语义分组

**问题**: 纯向量学习可能无法保证业务属性的严格区分

```
场景: 需要严格区分免费和付费内容

纯VQ可能出现:
  免费电影A → <a_12><b_83><c_4><d_701><e_25>
  付费电影B → <a_12><b_83><c_4><d_701><e_25>  (碰撞!)

HardCode保证:
  免费电影A → <a_12><b_83><c_4><d_701><e_49>  (saletype=1)
  付费电影B → <a_12><b_83><c_4><d_701><e_42>  (saletype=2)
```

### 动机2: 便于业务过滤

**优势**: 可以直接根据L5索引过滤结果

```python
# 只推荐免费内容
sids = generate_sids(query)
free_sids = [sid for sid in sids if sid.split('<e_')[1].startswith('49')]

# 或在Trie约束时直接限制
valid_next_tokens = trie.get_valid_next(prefix)
valid_next_tokens = [t for t in valid_next_tokens if is_free(t)]
```

### 动机3: 节省码本容量

**问题**: saletype只有2-3个取值，不需要完整的256个码位

```
传统VQ L5 (256码本):
  - 可能只用到20-30个码位
  - 其余200+码位浪费

HardCode L5:
  - saletype占用2-3个码位
  - 可以和publish_year组合
  - 理论上可表达 4(saletype) × 128(year) = 512种组合
```

---

## HardCodeMapper实现

### 类定义

```python
class HardCodeMapper(nn.Module):
    """Pack sale_type (0-3) and publish_year_bucket (0-127) into a single codebook index.

    编码公式:
        index = (publish_year_bucket & 0x7F) << 2 | (sale_type & 0x3)
    
    参数:
        bits: 索引位数，默认9位（可表达0-511）
    """
    
    def __init__(self, bits: int = 9):
        super().__init__()
        self.bits = bits
        self.max_index = (1 << bits) - 1  # 2^9 - 1 = 511
```

### 字段解析

#### 1. sale_type 解析

```python
@staticmethod
def _parse_sale_type(v) -> int:
    """解析付费类型
    
    支持的输入格式:
        - "FREE" / "free" → 1
        - "PAY" / "pay"   → 2
        - "0" / 0         → 0
        - "1" / 1         → 1
        - "2" / 2         → 2
        - None            → 0 (默认)
    
    返回: 0-3 的整数
    """
    if v is None:
        return 0
    
    if isinstance(v, str):
        s = v.strip()
        
        # 如果是数字字符串，直接转int
        if s.isdigit():
            return int(s)
        
        # 如果是文本，转大写匹配
        u = s.upper()
        if u == "FREE":
            return 1
        if u == "PAY":
            return 2
        return 0
    
    # 尝试直接转int
    try:
        return int(v)
    except Exception:
        return 0
```

**兼容的字段名**:
- `saletype`
- `sale_type`
- `pay_type`
- `paytype`

#### 2. publish_year 解析

```python
# 年份 → bucket转换
year = 2023
bucket = 2035 - year  # 2035 - 2023 = 12

# 限制范围 [0, 125]
if bucket >= 0 and bucket <= 125:
    bucket = bucket
else:
    bucket = 0  # 超出范围视为未知
```

**bucket含义**:
```
2035年 → bucket 0
2034年 → bucket 1
2033年 → bucket 2
...
2023年 → bucket 12
...
1910年 → bucket 125
<1910年 → bucket 0 (视为未知)
>2035年 → bucket 0 (视为未知)
```

### 编码公式

```python
# 位操作编码
index = (publish_year_bucket & 0x7F) << 2 | (sale_type & 0x3)
```

**拆解说明**:

```
publish_year_bucket: 7位（0-127）
sale_type:          2位（0-3）

合并为9位索引:

  [bit 8-2: year_bucket] [bit 1-0: sale_type]
   --------7位---------    -----2位-----

示例1: year=2023, saletype="PAY"
  bucket = 2035 - 2023 = 12 = 0b0001100
  sale   = 2            = 0b10
  index  = 0b000110010  = 50

示例2: year=2025, saletype="FREE"
  bucket = 2035 - 2025 = 10 = 0b0001010
  sale   = 1            = 0b01
  index  = 0b000101001  = 41

示例3: year=2019, saletype="PAY"
  bucket = 2035 - 2019 = 16 = 0b0010000
  sale   = 2            = 0b10
  index  = 0b001000010  = 66
```

### lookup方法

```python
def lookup(self, fields: dict, x_shape: torch.Size, device, dtype):
    """根据业务字段查找硬编码索引
    
    参数:
        fields: 包含业务字段的字典
            例: {"saletype": ["2", "1", "2"], "publish_year": ["2023", "2024", "2025"]}
        x_shape: 输出向量的形状 (batch, e_dim)
        device: 设备
        dtype: 数据类型
    
    返回:
        vec: 零向量 (batch, e_dim) - 硬编码不参与重构
        idx: 索引 (batch, 1)
    """
    batch = x_shape[0]
    
    # 1. 提取字段
    sale_vals = fields.get("saletype") or fields.get("sale_type") or ...
    publish_vals = fields.get("publish_year")
    
    # 2. 补齐batch
    sale_vals = self._pad_to_batch(sale_vals, batch, fill=0)
    publish_vals = self._pad_to_batch(publish_vals, batch, fill=0)
    
    # 3. 解析为整数
    st = torch.tensor([self._parse_sale_type(v) for v in sale_vals])
    year = torch.tensor([self._parse_int(v, 0) for v in publish_vals])
    
    # 4. 计算bucket
    bucket = 2035 - year
    bucket = torch.where((bucket >= 0) & (bucket <= 125), bucket, 0)
    
    # 5. 编码
    idx_val = ((bucket & 0x7F) << 2) | (st & 0x3)
    idx_val = torch.clamp(idx_val, max=self.max_index)  # max=511
    idx = idx_val.view(batch, 1)
    
    # 6. 向量为零（硬编码不参与重构）
    vec = torch.zeros(x_shape, device=device, dtype=dtype)
    
    return vec, idx
```

---

## 编码规则详解

### 完整编码映射表（部分）

| year | saletype | bucket | sale | index计算 | 最终index |
|---|---|---:|---:|---|---:|
| 2035 | FREE | 0 | 1 | (0<<2) \| 1 | **1** |
| 2035 | PAY | 0 | 2 | (0<<2) \| 2 | **2** |
| 2034 | FREE | 1 | 1 | (1<<2) \| 1 | **5** |
| 2034 | PAY | 1 | 2 | (1<<2) \| 2 | **6** |
| 2025 | FREE | 10 | 1 | (10<<2) \| 1 | **41** |
| 2025 | PAY | 10 | 2 | (10<<2) \| 2 | **42** |
| 2023 | FREE | 12 | 1 | (12<<2) \| 1 | **49** |
| 2023 | PAY | 12 | 2 | (12<<2) \| 2 | **50** |
| 2019 | PAY | 16 | 2 | (16<<2) \| 2 | **66** |
| 1910 | FREE | 125 | 1 | (125<<2) \| 1 | **501** |
| 1910 | PAY | 125 | 2 | (125<<2) \| 2 | **502** |

### 解码公式

```python
def decode_index(index: int):
    """从索引反推业务字段"""
    sale_type = index & 0x3  # 低2位
    year_bucket = (index >> 2) & 0x7F  # 高7位
    
    year = 2035 - year_bucket
    
    sale_name = {0: "UNKNOWN", 1: "FREE", 2: "PAY", 3: "RESERVED"}[sale_type]
    
    return year, sale_name

# 示例
decode_index(49)  # → (2023, "FREE")
decode_index(50)  # → (2023, "PAY")
decode_index(42)  # → (2025, "PAY")
```

### 索引分布分析

```python
# 理论上9位可表达 0-511 (512种)
# 实际使用:
#   year: 125个bucket (1910-2035)
#   sale: 4种 (0=UNKNOWN, 1=FREE, 2=PAY, 3=RESERVED)
# 组合: 125 × 4 = 500种有效组合

# 常见索引范围（假设主要是2015-2025年的数据）:
year_range = range(2015, 2026)  # 11年
buckets = [2035 - y for y in year_range]  # [10, 11, ..., 20]

# 对应索引:
for bucket in buckets:
    free_idx = (bucket << 2) | 1
    pay_idx = (bucket << 2) | 2
    print(f"Year {2035-bucket}: FREE={free_idx}, PAY={pay_idx}")

# 输出:
# Year 2025: FREE=41, PAY=42
# Year 2024: FREE=45, PAY=46
# Year 2023: FREE=49, PAY=50
# Year 2022: FREE=53, PAY=54
# ...
```

---

## 在RQVAE中的使用

### 初始化

```python
class RQVAE(nn.Module):
    def __init__(self, ...):
        # L1-L4: 学习的残差量化
        self.rq = ResidualVectorQuantizer(
            self.num_emb_list[:4],  # [1024, 1024, 1024, 1024]
            self.e_dim,
            ...
        )
        
        # L5: 硬编码
        self.hardcode = HardCodeMapper()
        
        # 解码器
        self.decoder = MLPLayers(...)
```

### 前向传播

```python
def forward(self, x: torch.Tensor, use_sk: bool = True, hard_fields=None):
    # 1. 编码
    x_e = self.encoder(x)  # (batch, e_dim)
    
    # 2. L1-L4 残差量化
    x_q, rq_loss, indices = self.rq(x_e, use_sk=use_sk)
    # x_q: (batch, e_dim) - L1-L4的量化向量之和
    # indices: (batch, 4) - L1-L4的索引
    
    # 3. L5 硬编码
    if hard_fields is not None:
        # 查找硬编码向量和索引
        hard_vec, hard_idx = self.hardcode.lookup(
            hard_fields, 
            x_q.shape, 
            x_e.device, 
            x_e.dtype
        )
        # hard_vec: (batch, e_dim) - 全零向量
        # hard_idx: (batch, 1) - 硬编码索引
        
        # 累加量化向量（实际上加0）
        x_q = x_q + hard_vec
        
        # 拼接索引
        indices = torch.cat([indices, hard_idx], dim=-1)
        # indices: (batch, 5) - L1-L5的索引
    else:
        # 如果没有提供hard_fields，L5索引为0
        hard_idx = torch.zeros((indices.shape[0], 1), device=x_q.device, dtype=indices.dtype)
        indices = torch.cat([indices, hard_idx], dim=-1)
    
    # 4. 解码
    out = self.decoder(x_q)  # (batch, in_dim)
    
    return out, rq_loss, indices
```

### 推理（获取索引）

```python
@torch.no_grad()
def get_indices(self, xs: torch.Tensor, use_sk: bool = False, hard_fields=None):
    # 1. 编码
    x_e = self.encoder(xs)
    
    # 2. L1-L4量化
    _, _, indices = self.rq(x_e, use_sk=use_sk)
    
    # 3. L5硬编码
    if hard_fields is not None:
        _, hard_idx = self.hardcode.lookup(hard_fields, x_e.shape, x_e.device, x_e.dtype)
        indices = torch.cat([indices, hard_idx], dim=-1)
    else:
        hard_idx = torch.zeros((indices.shape[0], 1), device=indices.device, dtype=indices.dtype)
        indices = torch.cat([indices, hard_idx], dim=-1)
    
    return indices  # (batch, 5)
```

---

## 训练和推理流程

### 训练时

```python
# 训练脚本
for epoch in range(epochs):
    for batch in dataloader:
        embeddings = batch['embedding']  # (batch, 2560)
        
        # ⚠️ 训练时通常不提供hard_fields
        # L5索引全部为0
        out, rq_loss, indices = model(embeddings, use_sk=True, hard_fields=None)
        
        # 只有L1-L4参与loss计算
        loss, recon_loss = model.compute_loss(out, rq_loss, embeddings)
        
        # 反向传播（只更新L1-L4和encoder/decoder）
        loss.backward()
        optimizer.step()
```

**关键点**:
- 训练时L5不参与，hard_fields=None
- L5的索引始终为0
- 只有L1-L4通过梯度学习

### 生成索引时

```python
# 索引生成脚本
import json

# 加载训练好的模型
model.load_state_dict(torch.load(checkpoint))
model.eval()

# 加载数据
embeddings = np.load('embeddings.npy')
with open('embeddings_meta.jsonl') as f:
    metadata = [json.loads(line) for line in f]

# 批量生成索引
results = []
for i in range(0, len(embeddings), batch_size):
    batch_emb = torch.tensor(embeddings[i:i+batch_size])
    batch_meta = metadata[i:i+batch_size]
    
    # ✅ 推理时提供hard_fields
    hard_fields = {
        'saletype': [m['pay_type'] for m in batch_meta],
        'publish_year': [m['publish_year'] for m in batch_meta]
    }
    
    # 获取索引（包含L5的真实硬编码值）
    indices = model.get_indices(batch_emb, use_sk=False, hard_fields=hard_fields)
    
    # 保存
    for j, idx in enumerate(indices):
        results.append({
            'idx': i + j,
            'indices': idx.tolist(),  # [12, 83, 4, 701, 50]
            'tokens': [f'<{chr(97+k)}_{idx[k]}>' for k in range(5)],
            'saletype': hard_fields['saletype'][j]
        })

# 输出
with open('indices.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
```

**关键点**:
- 推理时必须提供hard_fields
- L5索引根据实际业务字段计算
- 相同的pay_type和publish_year总是得到相同的L5索引

---

## 实际效果分析

### 索引分布

以193,034条视频数据为例：

```python
# 统计L5索引分布
l5_indices = [r['indices'][4] for r in all_results]
counter = Counter(l5_indices)

# 结果（示例）:
# index 50 (2023, PAY):  85,234条 (44.2%)
# index 49 (2023, FREE): 42,156条 (21.8%)
# index 46 (2024, PAY):  28,901条 (15.0%)
# index 45 (2024, FREE): 15,432条 (8.0%)
# index 42 (2025, PAY):  12,345条 (6.4%)
# ...
# 其他索引:              8,966条 (4.6%)
```

### 利用率

```python
# L5理论容量: 256 (原始设计) 或 512 (9-bit)
# 实际使用: 约20-40个索引

utilization = len(counter) / 256 * 100
# 约 7.8% - 15.6%
```

**问题**: L5利用率低

**原因**:
1. 数据集中的year分布集中（主要是2020-2025）
2. saletype只有2种（FREE/PAY）
3. 组合后实际只用到少数索引

### 与纯VQ对比

| 指标 | 纯VQ L5 | HardCode L5 |
|---|---|---|
| 训练参数 | ✅ 学习 | ❌ 固定规则 |
| 利用率 | 可能更高 | 较低（7-15%） |
| 语义连贯性 | 数据驱动 | 强制分组 |
| 业务过滤 | 需依赖前4层 | ✅ 直接判断 |
| 跨域泛化 | ✅ 更好 | ❌ 绑定业务 |
| 解释性 | 较弱 | ✅ 强 |

---

## 总结

### HardCode构建要点

1. **不参与训练**: L5的映射规则是预定义的，不通过梯度学习
2. **位操作编码**: 用位运算将多个字段打包到一个整数索引
3. **零向量设计**: 硬编码层的量化向量设为0，不影响重构
4. **推理时启用**: 训练时L5索引为0，推理时根据实际字段计算

### 适用场景

✅ **推荐使用HardCode**:
- 需要严格区分业务属性（免费/付费）
- 需要直接根据属性过滤结果
- 数据集属性分布明确

❌ **不推荐使用HardCode**:
- 追求最大语义表达能力
- 需要跨域泛化
- 业务字段可能频繁变化

### 替代方案

如果不使用HardCode（如RQ-OPQ）:
- L5完全由向量量化学习
- 所有512个码位用于语义
- 业务属性需依赖前4层隐式编码
- 灵活性更强，泛化能力更好

---

**参考代码**: [sid/src/models/rqvae.py](../sid/src/models/rqvae.py)  
**最后更新**: 2024-01-01
