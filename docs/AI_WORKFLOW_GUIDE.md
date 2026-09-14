# AI 协作迭代工作流指南

这份文档解决你提到的两个核心问题：
1. **对话越来越长，模型记忆错乱** → 用文件记忆 + 短对话
2. **每次都在同一份代码上改** → 用 Git 分支隔离版本

---

## 核心原则

### 1. 对话是工作台，文件是记忆

- **对话**：临时工作空间，用完就归档
- **文件**：永久记录，`AGENTS.md` 是项目的长期记忆

**不要**让一个对话承载整个项目历史。应该：
- 每个功能/修复开一个新对话
- 对话只做一件相对完整的事
- 完成后把关键信息写入文件，归档对话

### 2. 用 Git 保护每一版代码

- 每次改动前，工作区保持干净 (`git status` 无未提交内容)
- 每个迭代用独立分支
- 小步提交，一个 commit 对应一个明确改动
- 改坏了？`git reset` 或 `git checkout .` 立刻恢复

---

## 标准迭代流程

### 开始新迭代前

```powershell
# 1. 确保当前分支干净
git status
# 如果有未提交内容，决定是提交还是丢弃

# 2. 切回主分支，拉取最新代码
git checkout main
git pull origin main

# 3. 创建新分支 (命名：功能简写/日期/描述)
git checkout -b feat/add-encoder
# 或者修复：fix/data-loader-bug
```

### 与 Codex 协作

#### A. 开新对话

不要在老对话里继续聊。在 Codex 里创建新任务，第一条消息包含：

1. **当前目标** (一句话)
2. **起始状态** (在哪个分支/commit)
3. **关键约束** (什么不能动)
4. **具体任务** (分点列出)

可以用 `docs/ITERATION_TEMPLATE.md` 模板快速填写。

**示例**:
```
目标：实现 RQVAE 的 encoder 模块

起始分支：feat/add-encoder (刚从 main 切出)
关键约束：
- 使用 PyTorch
- 输入格式是 (B, C, H, W)
- 不改动 decoder 部分

任务：
1. 在 models/encoder.py 实现 Encoder 类
2. 添加单元测试
3. 更新 AGENTS.md 的项目结构说明
```

#### B. Codex 工作期间

Codex 会自动读取 `AGENTS.md`，获取项目记忆。你不需要重复解释项目背景。

**及时提交**：
- Codex 完成一个子任务后，立刻 `git add` + `git commit`
- commit message 写清楚改了什么
- 这样随时可以回退到任意一步

```powershell
# Codex 改完代码后
git add models/encoder.py tests/test_encoder.py
git commit -m "feat: implement RQVAE encoder with residual blocks"
```

#### C. 迭代结束时

1. **验证代码**
```powershell
# 运行测试
pytest tests/

# 或运行训练脚本验证
python train.py --config configs/test.yaml
```

2. **更新项目记忆**

编辑 `AGENTS.md`：
- 更新"当前状态"部分
- 记录新增功能
- 记录已知问题

3. **合并到主分支**
```powershell
# 切回 main 并合并
git checkout main
git merge feat/add-encoder

# 推送到远程
git push origin main

# 删除已合并的分支 (可选)
git branch -d feat/add-encoder
```

4. **归档对话**

在当前 Codex 对话里，让它产出一份交接说明：

```
这次迭代完成了，帮我生成交接说明，包含：
1. 改了哪些文件
2. 为什么这么改
3. 遗留问题 / 下一步建议

格式用 docs/ITERATION_TEMPLATE.md 的"迭代结束"部分
```

保存输出到 `docs/iterations/2026-09-14-encoder.md`。

然后归档这个 Codex 任务，不要再在里面继续新需求。

---

## 出问题时的救援方案

### 问题 1：改坏了，想回退

**场景**：Codex 改了代码，运行出错，不知道哪里坏了

**方案**：
```powershell
# 如果改动还没提交，直接丢弃
git checkout .

# 如果已经提交，回退到上一个提交
git reset --hard HEAD~1

# 如果已经推送到远程，需要强制推送 (谨慎)
git push origin main --force
```

### 问题 2：对话太长，模型开始胡说

**症状**：Codex 开始引用不存在的函数、混淆不同版本的代码

**方案**：
1. **立刻停止**当前对话
2. 用 `git status` 和 `git diff` 看清楚当前改了什么
3. 决定：
   - 改动合理 → 提交 → 开新对话继续
   - 改动有问题 → `git checkout .` 丢弃 → 开新对话重新来
4. 新对话里**明确告诉 Codex 当前状态**，不要让它依赖旧对话记忆

### 问题 3：不知道改了什么

**方案**：
```powershell
# 查看改动的文件列表
git status

# 查看具体改了什么内容
git diff

# 查看已提交的历史
git log --oneline -10

# 查看某次提交的具体改动
git show <commit-hash>
```

---

## 进阶技巧

### 1. 用标签标记里程碑

```powershell
# 达到某个重要节点时 (比如第一版训练通过)
git tag -a v0.1 -m "First working training loop"
git push origin v0.1
```

### 2. 实验分支

不确定的改动，用实验分支：

```powershell
git checkout -b experiment/new-loss-function
# 尝试新想法...
# 如果成功，合并回 main
# 如果失败，直接删除分支
git checkout main
git branch -D experiment/new-loss-function
```

### 3. 保存未完成的工作

需要临时切换任务：

```powershell
# 保存当前工作状态 (不提交)
git stash push -m "half-done refactoring"

# 切换到其他分支处理紧急事项
git checkout main
# ...

# 回来继续
git checkout feat/refactor
git stash pop
```

---

## 检查清单

开始新迭代前：
- [ ] 工作区干净 (`git status` 无未提交内容)
- [ ] 创建了新分支
- [ ] 开了新 Codex 对话
- [ ] 第一条消息包含：目标、起始状态、约束、任务

迭代结束时：
- [ ] 代码通过测试/验证
- [ ] 改动已提交 (commit message 清晰)
- [ ] `AGENTS.md` 已更新
- [ ] 生成了交接说明文档
- [ ] 分支已合并到 main
- [ ] Codex 对话已归档

---

## 常见反模式 (不要这么做)

❌ **在一个对话里做多个不相关功能**
→ 对话会失控，模型会混淆

❌ **不提交就开始下一个改动**
→ 改坏了无法回退，diff 混在一起看不清

❌ **直接在 main 分支上改**
→ 出问题时影响整个项目，无法隔离风险

❌ **对话太长还不开新任务**
→ 模型会开始引用不存在的东西，越改越乱

❌ **不更新 AGENTS.md**
→ 下次开新对话，Codex 不知道项目当前状态，重复解释浪费时间

---

## 总结

记住三句话：

1. **对话短，文件记** — `AGENTS.md` 是项目记忆，对话只是临时工作台
2. **分支隔离，小步提交** — 每个改动都能回退，永远不怕改坏
3. **一次一事，完成归档** — 不要让一个对话背负整个项目

按这个流程，模型记忆错乱和代码难以回退的问题就彻底解决了。
