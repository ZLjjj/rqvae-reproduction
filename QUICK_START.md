# 快速参考：AI 协作工作流

## 3 步开启新迭代

### 1️⃣ 准备分支
```powershell
git checkout main
git pull origin main
git checkout -b feat/你的功能名
```

### 2️⃣ 创建新 Codex 对话

第一条消息：
```
## 迭代信息

**目标**: [一句话说清楚]

**分支**: feat/你的功能名

**约束**: [什么不能动]

**任务**:
1. [具体任务 1]
2. [具体任务 2]

**验收标准**: [怎么算完成]
```

参考完整模板：`docs/ITERATION_TEMPLATE.md`

### 3️⃣ 迭代结束
```powershell
# 验证改动
git status
git diff

# 提交
git add .
git commit -m "feat: 你的改动说明"

# 合并回主分支
git checkout main
git merge feat/你的功能名
git push origin main

# 更新项目记忆
# → 编辑 AGENTS.md，更新"当前状态"部分
```

---

## 遇到问题

| 问题 | 解决 |
|-----|-----|
| 改坏了 | `git checkout .` (丢弃) 或 `git reset --hard HEAD~1` (回退) |
| 对话太长，模型胡说 | 停止 → 提交现有改动 → 开新对话 |
| 忘了改了什么 | `git diff` 或 `git status` |
| 需要临时切换任务 | `git stash push -m "说明"` 保存，后面 `git stash pop` 恢复 |

---

## 文件说明

- **`AGENTS.md`** — 项目长期记忆，Codex 会自动读取。每次迭代结束更新。
- **`docs/AI_WORKFLOW_GUIDE.md`** — 详细工作流指南，遇到问题时参考。
- **`docs/ITERATION_TEMPLATE.md`** — 新迭代前填这个模板，作为 Codex 的起始消息。
- **`QUICK_START.md`** — 这个文件，快速参考。

---

## 核心原则

✅ 对话是工作台，文件是记忆 → 用 `AGENTS.md` 记录项目状态，不靠对话记忆

✅ 用分支隔离版本 → 每个功能/修复独立分支，改坏了能回退

✅ 短对话 + 小提交 → 一个对话一件事，一个 commit 一个改动

✅ 完成后归档 → 产出交接说明，关闭对话，开新对话继续下一个任务

---

更详细的说明和进阶技巧，见 `docs/AI_WORKFLOW_GUIDE.md`
