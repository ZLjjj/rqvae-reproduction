# AI 迭代开发历史

> 本文件记录在完整 RQVAE 项目基础上进行的每次 AI 辅助改进，包含需求思路和实现内容。

---

## 版本跟踪说明

- **基准版本**: GitHub main 分支 (commit 7ce5a71)
- **开发分支**: dev-iteration
- **记录格式**: 每次迭代包含版本号、日期、需求思路、实现内容、改动文件、提交ID
- **工作流**: 开发完成后提交到 dev-iteration，然后推送到 GitHub

---

## Baseline - 2026-09-14
**项目名称:** GenSearchRec - RQVAE Reproduction Project  
**基准提交:** 7ce5a71 fix: 将L5层容量从256扩大到512以支持更大年份范围

**项目概述:**
基于多种量化策略（RVQ/FSQ/RQ-OPQ）的自编码模型，对高维嵌入进行离散压缩与重建，适用于搜索/推荐场景的嵌入压缩、离线编码与重建评估。

**核心模块:**
- **sid/** - 核心SID量化模块
  - models/ - RQVAE, FSQVAE, RQ-OPQ-VAE 等模型实现
  - scripts/ - 训练、评估、embedding生成脚本
  - src/ - codebook、embedding处理工具
  
- **sft/** - SFT微调模块
  - datasets/ - 数据集生成工具
  - eval/ - 推理与评估脚本
  
- **knowledge/** - 实验知识库
  - decisions.md - 决策记录
  - successful-runs.md - 成功实验记录
  - failed-runs.md - 失败实验记录

**关键文档:**
- README.md - 项目总览
- REPRODUCTION_GUIDE.md - 完整复现指南
- RQVAE_L1_SINKHORN_README.md - L1 Sinkhorn方法说明
- SID_THREE_WAY_RUNBOOK.md - SID三方对比实验手册

---

## v1 - 2026-09-14
**需求思路:** 为完整 RQVAE 项目建立 AI 迭代版本管理系统，支持长期开发

**实现内容:**
- 基于 GitHub main 分支创建 dev-iteration 开发分支
- 添加 ITERATIONS.md 版本跟踪文件
- 记录项目基准状态和模块结构
- 建立清晰的迭代规范

**改动文件:** 
- ITERATIONS.md (新增)

**提交ID:** c2f2c75

---

## 下一步迭代说明

当需要保存版本时，在 ITERATIONS.md 中添加如下内容：

```markdown
## v{N} - {日期}
**需求思路:** {描述本次的需求和思路}

**实现内容:**
- {实现项1}
- {实现项2}

**改动文件:** 
- {文件1}
- {文件2}

**提交ID:** {git commit hash}
```

然后执行：
```bash
git add ITERATIONS.md {改动的文件}
git commit -m "v{N}: {简短描述}

需求思路: {详细描述}
实现内容: {实现清单}
"
git push origin dev-iteration
```
