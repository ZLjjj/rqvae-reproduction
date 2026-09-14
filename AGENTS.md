# RQVAE Reproduction Project

## 项目概述

这是一个 RQVAE (Residual-Quantized Variational AutoEncoder) 的复现项目。

**技术栈**:
- Python 3.x
- PyTorch
- 深度学习相关库 (根据实际使用补充)

## 项目结构

```
new_pro/
├── models/          # 模型定义
├── data/            # 数据加载和预处理
├── training/        # 训练脚本
├── utils/           # 工具函数
├── configs/         # 配置文件
├── checkpoints/     # 模型检查点 (git ignored)
└── experiments/     # 实验记录
```

## 核心约定

### 不要修改的部分

1. **模型架构的核心设计** - 除非明确要求重构，保持已验证的模型结构不变
2. **数据加载接口** - 保持数据 pipeline 的输入输出格式一致
3. **配置文件格式** - 维护配置文件的向后兼容性

### 代码规范

- 使用 type hints
- 函数和类添加 docstring
- 复杂逻辑添加注释说明
- 保持代码风格一致

### 实验管理

- 每个实验在 `experiments/` 下创建独立目录
- 记录超参数、训练曲线、最终指标
- 关键实验结果写入实验目录的 README

## AI 协作指南

### 当前状态

**阶段**: 开发中

**进度**: 基础框架已搭建，可以开始实现具体模块

**最近改动** (2026-09-14): 
- ✅ 创建了基础项目结构（models、configs、tests）
- ✅ 实现了 Encoder 模块（包含 ResidualBlock 和下采样层）
- ✅ 添加了配置文件模板（default.yaml）
- ✅ 编写了基础单元测试（test_encoder.py）
- ✅ 配置了依赖项（requirements.txt）

**已实现的模块**:
- `models/encoder.py`: 图像编码器
  - ResidualBlock：基础残差块
  - 两层下采样（4x 空间降维）
  - 输入: (B, 3, 256, 256) → 输出: (B, 512, 64, 64)

**下一步优先级**: 
1. 实现 Decoder 模块（与 Encoder 对称的上采样结构）
2. 实现 Residual Quantization 模块（多层量化器）
3. 实现完整的 RQVAE 模型（组合 Encoder、Quantizer、Decoder）
4. 添加训练脚本和数据加载器

**最后更新**: 2026-09-14

**当前版本**: 初始化

**已完成功能**:
- 项目结构初始化

**待实现功能**:
- 模型实现
- 数据加载
- 训练流程
- 评估脚本

**已知问题**:
- 无

### 依赖版本

<!-- 更新实际使用的版本 -->
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
# 根据实际需求补充
```

## 注意事项

- 模型训练前先在小数据集上验证流程
- 超参数调整通过配置文件，不硬编码
- 实验结果及时记录，避免重复工作
- checkpoint 命名包含关键信息 (epoch, metric)
