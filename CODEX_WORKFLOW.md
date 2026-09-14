# Codex 项目迭代工作流

> 在独立 Codex 项目中进行开发迭代，使用清晰的版本管理保持历史记录。

## 快速开始

### 1. 本地项目路径
```
C:\Users\dszlj\Desktop\new_pro
```

### 2. GitHub 仓库
```
https://github.com/ZLjjj/rqvae-reproduction
Branch: reproduce-rqvae (开发分支)
Branch: master (主分支，包含完整研究项目)
```

### 3. 版本记录文件
- **ITERATIONS.md** - 每次迭代的需求思路和实现内容
- **AGENTS.md** - AI 协作工作流文档

---

## 每次迭代的工作流

### 步骤1：在Codex里开发改进
在 Codex 项目中：
- 修改代码、添加功能
- 测试验证效果
- 整理思路

### 步骤2：保存版本（使用 $ai-iteration-tracker 技能）

**告诉助手需要保存版本时，提供：**
1. 本次的需求思路（为什么要这么改？）
2. 实现了哪些功能？
3. 改了哪些文件？

**助手会自动执行：**
```bash
# 更新 ITERATIONS.md
# 提交到本地 Git
git commit -m "v{N}: 版本描述

需求思路: ...
实现内容: ...
"

# 推送到 GitHub
git push origin reproduce-rqvae
```

### 步骤3：切换Codex窗口继续工作
- 其他 Codex 窗口可以随时查看最新代码
- 检查 ITERATIONS.md 了解最新改动
- 基于最新版本继续开发

---

## 版本对比和回退

### 查看迭代历史
```bash
git log --oneline --graph
# 或查看 ITERATIONS.md
```

### 对比两个版本
```bash
git diff v1 v2
```

### 回到某个历史版本（需明确声明）
```bash
# 查看而不修改
git show {commit-hash}

# 创建分支从历史版本重新开始
git checkout -b new-attempt {commit-hash}
```

---

## 关键文件

| 文件 | 用途 |
|------|------|
| ITERATIONS.md | 记录每个版本的需求和实现 |
| AGENTS.md | AI 协作指南 |
| QUICK_START.md | 项目快速参考 |
| configs/default.yaml | 项目配置 |
| models/encoder.py | 核心代码 |
| tests/test_encoder.py | 测试框架 |
| requirements.txt | 依赖列表 |

---

## 技能速查

当需要以下操作时，告诉助手：

**保存版本**
> "需要保存这个版本"
> 提供：需求思路、实现内容、改动文件

**查看历史**
> "查看迭代历史"
> "对比v1和v2"

**回到某版本**
> "回到v3版本"
> (助手会明确说明这是什么操作)

---

## 最佳实践

✅ **推荐做法**
- 每个有意义的改动都保存为一个版本
- 保持 ITERATIONS.md 的最新性
- 每个版本配一行简洁的需求描述

❌ **避免做法**
- 一份代码上反复修改而不保存版本
- 忘记记录改动思路
- 长对话后还没提交就创建新Codex窗口

---

## 协作场景

### 场景1：多个Codex窗口迭代
```
Codex窗口1: 做功能A → 保存v1
    ↓ (推送到GitHub)
Codex窗口2: 拉取最新 → 基于v1做功能B → 保存v2
    ↓ (推送到GitHub)
Codex窗口1: 拉取最新 → 继续改进 → 保存v3
```

### 场景2：发现问题需要回退
```
v1 ✅ 工作正常
v2 ✅ 添加新功能
v3 ❌ 引入了bug
    → 创建分支从v2重新开始
    → v3-fix 修复问题
```
