from pathlib import Path
root = Path(__file__).parent
src = (root / 'sid_three_way_feishu.md').read_text(encoding='utf-8')
app = '''
---

## 码本利用率补充

| 方案 | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| baseline 视频 | 41.02% | 84.77% | 63.18% | 98.24% | 45.51% |
| 任务2 同源 video-only | 4.79% | 100.00% | 100.00% | 100.00% | 45.51% |
| 任务3 cleaned text_format | 7.71% | 100.00% | 100.00% | 100.00% | 45.51% |

注：码本利用率按“该层实际使用的 unique code 数 / 该层 codebook 容量”计算；L1-L4 容量为 1024，L5 按 9-bit hard-code 容量 512 计算。该指标反映每层 codebook 的覆盖程度，与完整 SID 独立编码率不同。
'''
(root / 'sid_three_way_feishu_corrected.md').write_text(src + app, encoding='utf-8')
print(root / 'sid_three_way_feishu_corrected.md')
