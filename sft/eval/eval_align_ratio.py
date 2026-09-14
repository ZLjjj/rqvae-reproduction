#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 未更新，直接用align_infer_and_eval.py
import json
import re
from collections import defaultdict

dataname = "merged_emb.npy_rqopq_base_3rq_1opq2_bigemb"

INFER_PATH = f"/lizhaoxuan1/llm_output/align_infer_constrained_train_Base.jsonl"
SID_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"


# 相关性判断
def norm_text(s: str) -> str:
    if s is None:
        return ""
    if isinstance(s, (dict, list)):
        try:
            s = json.dumps(s, ensure_ascii=False, sort_keys=True)
        except Exception:
            s = str(s)
    if not isinstance(s, str):
        s = str(s)

    s = s.strip()
    # 去掉多余空白
    s = re.sub(r"\s+", " ", s)
    # 去掉末尾标点
    s = s.strip(" \t\r\n,，。；;")
    return s

def match_top1(label: str, pred: str) -> bool:
    label_n = norm_text(label)
    pred_n = norm_text(pred)
    if not label_n or not pred_n:
        return False
    if label_n == pred_n:
        return True
    # 包含关系（任一方包含另一方）
    if label_n in pred_n or pred_n in label_n:
        return True
    return False

def res_match_item(label: str, pred: str):
    partten = r""
    return False

def match_sid(task: str, label: str, pred: str, preds: list, sid_map: dict):
    if not label or not pred:
        return False

    label = label.strip()
    pred_sid = pred.strip()

    # 非法 SID
    if pred_sid not in sid_map:
        return -1

    if label == pred_sid:
        return 1

    meta_info = sid_map.get(pred_sid)
    if task == "tag_to_sid":
        label_tags = list(extract_tags(label))
        t = min(len(label_tags), 5)
        r = 0
        for tag in label_tags:
            if match_top1(tag, meta_info):
                r += 1
        if r/t > 0.5:
            return 1
        return 0
 
    if match_top1(label, meta_info):
        return 1

    return 0


# 抽取标签列表，逐tag计分
TAG_SPLIT_RE = re.compile(r"[、,，;/；\|\t\n]+")  # 支持多种分隔符

def extract_tags(text: str):
    """
    输入可能是：
      - "名称:xxx,资源类型:yyy,标签:aaa、bbb、ccc"
      - "标签:教育、成长、个人成长"
      - "教育、成长、个人成长"
    输出：去重后的 tag set
    """
    if not text:
        return set()

    t = text.strip()

    if "标签:" in t:
        t = t.split("标签:", 1)[1]

    t = t.split("\n")[0]
    t = t.strip(" ,，。；;")

    raw = [x.strip() for x in TAG_SPLIT_RE.split(t) if x.strip()]
    tags = set()
    for tag in raw:
        tag = tag.strip()
        if not tag:
            continue
        tags.add(tag)
    return tags

def tag_overlap_score(label_text: str, pred_text: str):
    """
    返回 (right_cnt, total_cnt)
    total_cnt：label 中 tag 数
    right_cnt：label 中每个 tag 是否在 pred tag set 中命中（命中则+1）
    """
    label_tags = list(extract_tags(label_text))
    pred_tags = extract_tags(pred_text)

    total_cnt = len(label_tags)
    right_cnt = 0
    for lt in label_tags:
        if lt in pred_tags:
            right_cnt += 1
    total_cnt = min(total_cnt, 5)
    right_cnt = min(right_cnt, total_cnt)
    return right_cnt, total_cnt

def load_sid_map(path):
    sid_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            sid = re.sub(r"[\[\]'，, ]", "", "".join(r.get("indices", "")))
            meta = r.get("json_format") or r.get("meta_json")
            if sid and meta:
                sid_map[sid] = meta
    return sid_map

def extract_text_label(text: str, label: str) -> str:
    if not text:
        return label

    # if not "从已学习过的有效SID中" in text:
    #     return label

    patterns = [
        r"名\s*:\s*(.*?)(?:，|$)",
        r"标签内容\s*:\s*(.*?)(?:，|$)",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

    return label

# ---------- 主统计 ----------
def main():
    sid_map = load_sid_map(SID_PATH)

    top1_stats = {
        "sid_to_item": {"right": 0, "total": 0},
        "sid_to_name": {"right": 0, "total": 0},
        "sid_to_category": {"right": 0, "total": 0},
        "sid_to_artist": {"right": 0, "total": 0},
        "item_to_sid": {"right": 0, "total": 0},
        "name_to_sid": {"right": 0, "total": 0},
        "category_to_sid": {"right": 0, "total": 0},
        "artist_to_sid": {"right": 0, "total": 0},
        "tag_to_sid": {"right": 0, "total": 0},
        "unknown": {"right": 0, "total": 0}
    }

    sid_val = {
        "item_to_sid": {"val": 0, "total": 0},
        "name_to_sid": {"val": 0, "total": 0},
        "category_to_sid": {"val": 0, "total": 0},
        "artist_to_sid": {"val": 0, "total": 0},
        "tag_to_sid": {"val": 0, "total": 0},
        "unknown": {"val": 0, "total": 0}
    }

    tag_stats = {"right_cnt": 0, "total_cnt": 0, "sample_total": 0}

    unknown = 0

    with open(INFER_PATH, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[SKIP] line {line_idx} json error: {e}")
                continue

            prompt = obj.get("prompt", "")
            label = obj.get("label", "")
            pred = obj.get("top1_predict", "")  # 兼容字段名
            preds =  obj.get("predicts", "")
            task = obj.get("task_type", "")
            if task is None:
                unknown += 1
                continue

            if task == "sid_to_tag":
                r, t = tag_overlap_score(label, pred)
                tag_stats["right_cnt"] += r
                tag_stats["total_cnt"] += t
                tag_stats["sample_total"] += 1
            # elif task == "sid_to_item":
            #     top1_stats[task]["total"] += 1
            #     if res_match_item(label, pred):
            #         top1_stats[task]["right"] += 1
            elif task.endswith("to_sid"):
                text_label = extract_text_label(prompt, label)
                top1_stats[task]["total"] += 1
                sid_val[task]["total"] += 1
                res = match_sid(task, text_label, pred, preds, sid_map)
                if res == 1:
                    top1_stats[task]["right"] += 1
                    sid_val[task]["val"] += 1
                elif res == 0:
                    sid_val[task]["val"] += 1
            else:
                top1_stats[task]["total"] += 1
                if match_top1(label, pred):
                    top1_stats[task]["right"] += 1

    # -------- 打印结果 --------
    def safe_acc(r, t):
        return (r / t) if t > 0 else 0.0

    print("========== Alignment Report ==========")
    for k in ["sid_to_name", "sid_to_category", "sid_to_artist"]:
        r = top1_stats[k]["right"]
        t = top1_stats[k]["total"]
        print(f"{k}: acc={safe_acc(r,t)*100:.2f}% ({r}/{t})")

    tr = tag_stats["right_cnt"]
    tt = tag_stats["total_cnt"]
    print(f"sid_to_tag: tag-level acc={safe_acc(tr,tt)*100:.2f}% (right_tags={tr} / total_tags={tt}), samples={tag_stats['sample_total']}")

    print("========== Alignment Report ==========")
    for k in ["item_to_sid", "name_to_sid", "tag_to_sid", "artist_to_sid"]:
        r = top1_stats[k]["right"]
        t = top1_stats[k]["total"]
        print(f"{k}: acc={safe_acc(r,t)*100:.2f}% ({r}/{t})")
    print("========== SID Valid Report ==========")
    for k in ["item_to_sid", "name_to_sid", "tag_to_sid", "artist_to_sid"]:
        r = sid_val[k]["val"]
        t = sid_val[k]["total"]
        print(f"{k}: val={safe_acc(r,t)*100:.2f}% ({r}/{t})")
    if unknown:
        print(f"[WARN] 未识别任务样本数: {unknown}")

if __name__ == "__main__":
    main()
