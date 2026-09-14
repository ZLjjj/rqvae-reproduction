#!/bin/bash

# ================= 配置区域 =================

# 脚本路径
SCRIPT_PATH="sid/rqvae.py"

# 数据路径 (请替换为实际生成的 .npy 文件路径)
# 示例: "data/my_embeddings.npy"
DATA_PATH="/mnt/zhanggehang1/Sid/gensearchrec/sid/emb/output_embeddings.npy"

# 检查点保存目录
CKPT_DIR="checkpoints/rqvae_run"

# 设备选择 (cuda:0, cuda:1, ... 或 cpu)
DEVICE="cuda:0"
# DEVICE="cpu"

# ================= 模型参数 =================

# MLP 层结构 (输入维度会自动适配数据，这里定义中间层)
LAYERS="1024 512 256 128"

# RVQ 参数
# num_emb_list: 每层的码本大小 (例如 3 层，每层 256 个聚类中心)
NUM_EMB_LIST="1024 1024 1024"
# e_dim: 码本向量的维度
E_DIM=1024

# Sinkhorn 温度参数 (对应 num_emb_list 的层数)
# 0.0 表示硬分配 (KMeans风格)，大于0表示软分配
SK_EPSILONS="0.003 0.003 0.003"

# ================= 训练参数 =================
EPOCHS=200
BATCH_SIZE=2048
LR=1e-3
EVAL_STEP=10

# ===========================================

# 确保脚本可执行
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: 找不到脚本文件 $SCRIPT_PATH"
    echo "请在项目根目录下运行此脚本。"
    exit 1
fi

echo "启动 RQVAE 训练..."
echo "数据路径: $DATA_PATH"
echo "设备: $DEVICE"

# 创建输出目录
mkdir -p "$CKPT_DIR"

python "$SCRIPT_PATH" \
    --data_path "$DATA_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --device "$DEVICE" \
    --layers $LAYERS \
    --num_emb_list $NUM_EMB_LIST \
    --e_dim "$E_DIM" \
    --sk_epsilons $SK_EPSILONS \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --eval_step "$EVAL_STEP" \
    --kmeans_init True \
    --loss_type "mse"

if [ $? -eq 0 ]; then
    echo "✅ 训练完成"
else
    echo "❌ 训练失败"
    exit 1
fi
