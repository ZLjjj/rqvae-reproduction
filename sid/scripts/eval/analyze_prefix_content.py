#!/usr/bin/env python3
"""Analyze raw content consistency within SID prefix buckets."""
import argparse, json, math, re
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ["video_type", "pay_type", "publish_year", "content_type", "areas", "languages", "genre_types", "llm_tag"]

def vals(v):
    if v is None: return []
    if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s: return []
    return [x.strip() for x in re.split(r"[;,；、|]+", s) if x.strip()]

def scalar(v):
    x = vals(v)
    return x[0] if x else "(缺失)"

def purity(records, field):
    c = Counter(scalar(r.get(field)) for r in records)
    return (c.most_common(1)[0][1] / len(records), c)

def jaccard(records, field):
    sets = [set(vals(r.get(field))) for r in records]
    sets = [x for x in sets if x]
    if len(sets) < 2: return None
    total = pairs = 0
    # deterministic cap for very large buckets
    if len(sets) > 256: sets = sets[:256]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            total += len(sets[i] & sets[j]) / len(u) if u else 1.0; pairs += 1
    return total / pairs if pairs else None

def prefix(tokens, level):
    vals = tokens or []
    return "".join(vals[:level])

def load(path, domain=None):
    groups = {i: defaultdict(list) for i in range(1, 6)}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if domain is not None and r.get("domain") != domain:
                continue
            if isinstance(r.get("meta"), dict):
                merged = dict(r["meta"])
                merged.update(r)
                r = merged
            toks = r.get("tokens") or r.get("sid")
            if not toks and r.get("indices"):
                toks = [f"<{chr(97+i)}_{v}>" for i,v in enumerate(r["indices"])]
            for level in range(1, 6): groups[level][prefix(toks, level)].append(r)
    return groups

def fmt(v):
    if v is None: return "不适用"
    return f"{v:.4f}"

def analyze(name, path, out, topk, domain=None):
    groups = load(path, domain); lines=[f"## {name}", "", f"输入：`{path}`", ""]
    for level in range(1, 5):
        gs = groups[level]; ordered = sorted(gs.items(), key=lambda kv: len(kv[1]), reverse=True)
        lines += [f"### L{level} 内容分析", "", f"共 {len(gs):,} 个 prefix bucket。下面列出样本量最大的 {min(topk,len(ordered))} 个 bucket。", ""]
        lines += ["| prefix | 样本数 | 主视频类型 purity | 主题材 purity | 付费状态 purity | 年份 purity | genre Jaccard | 代表内容 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for pfx, rs in ordered[:topk]:
            pu, _ = purity(rs,"video_type"); pg,_=purity(rs,"genre_types"); pp,_=purity(rs,"pay_type"); py,_=purity(rs,"publish_year")
            gj=jaccard(rs,"genre_types")
            sample=[]
            for r in rs[:3]:
                title=r.get("meta_main_name") or r.get("meta_origin_name") or r.get("id")
                sample.append(str(title).replace("|","/").replace("\n"," ")[:55])
            lines.append(f"| `{pfx}` | {len(rs):,} | {pu:.2%} | {pg:.2%} | {pp:.2%} | {py:.2%} | {fmt(gj)} | {'；'.join(sample)} |")
        lines += ["", "字段分布示例："]
        for pfx, rs in ordered[:min(3,len(ordered))]:
            lines.append(f"- `{pfx}`：")
            for field in FIELDS:
                c=Counter(scalar(r.get(field)) for r in rs) if field not in ("areas","languages","genre_types","llm_tag") else Counter(x for r in rs for x in vals(r.get(field)))
                lines.append(f"  - {field}：" + "，".join(f"{k}({v})" for k,v in c.most_common(5)))
            lines.append("  - 样例：")
            for r in rs[:5]:
                lines.append(f"    - id={r.get('id')}；标题={r.get('meta_main_name') or r.get('meta_origin_name') or ''}；text_format={str(r.get('text_format',''))[:160]}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--baseline", required=True); p.add_argument("--task2", required=True); p.add_argument("--task3", required=True); p.add_argument("--output", required=True); p.add_argument("--topk", type=int, default=5); a=p.parse_args()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    tmp=[]
    for name,path,domain in [("baseline 视频",a.baseline,"video"),("任务2 同源 video-only",a.task2,None),("任务3 cleaned text_format",a.task3,None)]:
        part=out.parent/(out.stem+f"_{len(tmp)}.md"); analyze(name,path,part,a.topk,domain); tmp.append(part.read_text(encoding="utf-8"))
    out.write_text("# SID Prefix 原始内容一致性分析\n\n"+"\n\n---\n\n".join(tmp),encoding="utf-8")
    for x in tmp: pass
    print(out)

if __name__ == "__main__": main()
