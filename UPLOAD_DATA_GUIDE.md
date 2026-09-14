# 数据上传指南

> **创建时间**: 2026-09-14  
> **目的**: 帮助将GenSearchRec项目数据上传到GitHub

---

## 📊 当前数据情况

### 数据统计

```
总大小: 2154.65 MB (约2.15 GB)

results_rqvae_l1_sinkhorn/
├── task3/                    (1449.69 MB)
│   ├── checkpoints/          (683 MB, 17个.pth文件)
│   ├── indices/              (383.14 MB, indices.jsonl)
│   └── indices_final/        (383.14 MB, indices.jsonl)
└── task3_correct/            (704.96 MB)
    ├── checkpoints/          (321.6 MB, 8个.pth文件)
    └── indices/              (383.36 MB, indices.jsonl)
```

### 大文件清单（超过100MB）

⚠️ **GitHub单文件限制: 100 MB**

| 文件 | 大小 | 备注 |
|---|---:|---|
| task3/indices/indices.jsonl | 383.14 MB | ❌ 超限 |
| task3/indices_final/indices.jsonl | 383.14 MB | ❌ 超限 |
| task3_correct/indices/indices.jsonl | 383.36 MB | ❌ 超限 |
| *.pth (25个模型文件) | 40.2 MB × 25 | ✅ 可直接上传 |

---

## 🎯 推荐方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| **方案1: GitHub Release** ⭐ | 免费、简单、2GB限制 | 不支持版本控制 | 开源分享 |
| 方案2: Git LFS | 支持版本控制 | 免费1GB，超出付费 | 需要追踪变更 |
| 方案3: 精简版 | 仓库小 | 数据不完整 | 个人项目 |
| 方案4: 只上传代码 | 最小仓库 | 需要用户自己跑 | 学术论文 |

---

## 方案1: GitHub Release（推荐）

### 步骤

1. **压缩数据**
```bash
# 使用PowerShell压缩
Compress-Archive -Path results_rqvae_l1_sinkhorn -DestinationPath results_data.zip
```

2. **创建Release**
```bash
# 方法A: 使用GitHub CLI（推荐）
gh release create v1.0.0 results_data.zip --title "Training Results" --notes "完整训练数据"

# 方法B: 网页上传
# 访问 https://github.com/你的用户名/仓库名/releases/new
```

---

## 方案2: Git LFS

### 步骤

```bash
# 1. 初始化
git lfs install

# 2. 追踪大文件
git lfs track "*.jsonl"
git lfs track "*.pth"

# 3. 提交
git add .gitattributes results_rqvae_l1_sinkhorn/
git commit -m "Add training results"
git push origin main
```

---

## 快速命令

### 检查GitHub仓库
```bash
git remote -v
# 如果为空，需要先创建仓库并添加remote
```

### 添加远程仓库
```bash
git remote add origin https://github.com/你的用户名/gensearchrec.git
```

### 首次提交
```bash
git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main
```

