#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

from trie import SIDTrie
from constraint import SIDConstrainedLogitsProcessor

# dataname = "rqvae_opq_20260208_station_video_text_format_4B"
dataname = "vs_Qwen4B_rqvae_hc_ema"

MODEL_DIR = "/lizhaoxuan1/checkpoints/qwen3_4b_Instruct_sft_0417_sft_single_video_ext_vs_Qwen4B_rqvae_hc_ema_gpu24_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"
INPUT_PATH = f"/lizhaoxuan1/train_data/align_val_ext_station_0.95_video_0.99_{dataname}_v2.jsonl"
OUTPUT_PATH = "/lizhaoxuan1/llm_output/align_infer_constrained_train.jsonl"
SID_TRIE_PATH = f"/lizhaoxuan1/sid/result/sid_trie_{dataname}.pkl"
SID_MAPPING_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"

MAX_INFER_NUM = 4000
MAX_NEW_TOKENS = 16
NUM_RETURN = 5
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1

SID_TASKS = {
    "item_to_sid",
    "name_to_sid",
    "tag_to_sid",
    "artist_to_sid",
    "attr_to_sid",
    "ip_to_sid"
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_trie_from_sid_data(path):
    sids = set()
    pat = r"<a_(\d+)><b_(\d+)><c_(\d+)><d_(\d+)><e_(\d+)>"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                sid = "".join(r.get("sid", ""))
                m = re.match(pat, sid)
                if m:
                    sids.add(tuple(int(x) for x in m.groups()))
            except:
                pass
    max_vals = [0]*5
    for sid in sids:
        for i,v in enumerate(sid):
            max_vals[i] = max(max_vals[i], v)
    trie = SIDTrie(num_levels=5, codebook_sizes=[v+100 for v in max_vals])
    for sid in sids:
        trie.insert(list(sid))
    return trie

if os.path.exists(SID_TRIE_PATH):
    sid_trie = SIDTrie.load(SID_TRIE_PATH)
else:
    sid_trie = build_trie_from_sid_data(SID_MAPPING_PATH)
    sid_trie.save(SID_TRIE_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

eos_id = tokenizer.eos_token_id
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
eos_ids = [x for x in [eos_id, im_end_id] if x is not None]

sid_logits_processor = SIDConstrainedLogitsProcessor(
    trie=sid_trie,
    tokenizer=tokenizer,
    force_constraint=True,
    eos_token_id=eos_id,
    sep_token_id=None,
)

def drop_candidate_sid_list(t):
    return re.sub(r"从候选SID列表.*?中[,，]", "从已学习过的有效SID中", t, flags=re.DOTALL)

def build_inputs(prompt):
    text = tokenizer.apply_chat_template(
        [{"role":"user","content":prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").to(model.device)

def decode(seq, plen):
    return tokenizer.decode(seq[plen:], skip_special_tokens=True).strip()

def norm_text(s):
    if s is None:
        return ""
    if isinstance(s,(dict,list)):
        try:
            s = json.dumps(s,ensure_ascii=False,sort_keys=True)
        except:
            s = str(s)
    s = str(s).strip()
    s = re.sub(r"\s+"," ",s)
    return s.strip(" \t\r\n,，。；;")

def match_top1(l,p):
    l,p = norm_text(l), norm_text(p)
    if not l or not p: return False
    return l==p or l in p or p in l

TAG_SPLIT_RE = re.compile(r"[、,，;/；\|\t\n]+")

def extract_tags(t):
    if not t: return set()
    if "标签:" in t:
        t = t.split("标签:",1)[1]
    t = t.split("\n")[0].strip(" ,，。；;")
    return set(x for x in TAG_SPLIT_RE.split(t) if x.strip())

def tag_overlap_score(l,p):
    lt = list(extract_tags(l))
    pt = extract_tags(p)
    r = sum(1 for x in lt if x in pt)
    t = min(len(lt),5)
    return min(r,t), t

def load_sid_map(path):
    m = {}
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sid = re.sub(r"[\[\]'，, ]","", "".join(r.get("sid","")))
            meta = r.get("json_format") or r.get("meta_json")
            if sid and meta:
                m[sid]=meta
    return m

def extract_text_label(prompt,label):
    for p in [r"名\s*:\s*(.*?)(?:，|$)", r"标签内容\s*:\s*(.*?)(?:，|$)"]:
        m = re.search(p,prompt)
        if m:
            return m.group(1).strip()
    return label

def match_sid(task,label,pred,sid_map):
    if pred not in sid_map:
        return -1
    if label==pred:
        return 1
    meta = sid_map[pred]
    if task=="tag_to_sid":
        tags = extract_tags(label)
        if not tags: return 0
        hit = sum(1 for t in tags if match_top1(t,meta))
        t = min(len(tags), 5)
        return 1 if hit/t>0.5 else 0
    return 1 if match_top1(label,meta) else 0

sid_map = load_sid_map(SID_MAPPING_PATH)

top1_stats = defaultdict(lambda:{"right":0,"total":0})
sid_val = defaultdict(lambda:{"val":0,"total":0})
tag_stats = {"right_cnt":0,"total_cnt":0,"sample_total":0}

# ================= 新增：domain 维度统计 =================
domain_top1_stats = defaultdict(lambda: defaultdict(lambda: {"right": 0, "total": 0}))
domain_sid_val = defaultdict(lambda: defaultdict(lambda: {"val": 0, "total": 0}))
domain_tag_stats = defaultdict(lambda: {"right_cnt": 0, "total_cnt": 0, "sample_total": 0})
# ========================================================

with open(INPUT_PATH,"r",encoding="utf-8") as fin, open(OUTPUT_PATH,"w",encoding="utf-8") as fout:
    for idx,line in enumerate(fin):
        if idx > MAX_INFER_NUM:
            break
        try:
            d = json.loads(line)
        except:
            continue

        domain = d.get("domain", "unknown")
        task = d.get("task_type")

        msgs = d.get("messages",[])
        if not msgs or len(msgs)<2: continue
        prompt = drop_candidate_sid_list(msgs[0]["content"].strip())
        label = msgs[1]["content"].strip()

        inputs = build_inputs(prompt)
        plen = inputs["input_ids"].shape[1]
        lp = [sid_logits_processor] if task in SID_TASKS else None

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=NUM_RETURN,
                eos_token_id=eos_ids,
                pad_token_id=eos_id,
                repetition_penalty=REPETITION_PENALTY,
                logits_processor=lp,
            )

        preds = [decode(s,plen) for s in out]
        best = preds[0]

        fout.write(json.dumps({
            "domain": domain,
            "task_type":task,
            "idx":idx,
            "prompt":prompt,
            "label":label,
            "predicts":preds,
            "scores":[None]*len(preds),
            "top1_predict":best,
            "top1_score":None
        },ensure_ascii=False)+"\n")

        if task=="sid_to_tag":
            r,t = tag_overlap_score(label,best)
            tag_stats["right_cnt"]+=r
            tag_stats["total_cnt"]+=t
            tag_stats["sample_total"]+=1

            domain_tag_stats[domain]["right_cnt"] += r
            domain_tag_stats[domain]["total_cnt"] += t
            domain_tag_stats[domain]["sample_total"] += 1

        elif task and task.endswith("to_sid"):
            # tl = extract_text_label(prompt,label)
            tl = d.get("label") or label
            top1_stats[task]["total"]+=1
            sid_val[task]["total"]+=1

            domain_top1_stats[domain][task]["total"] += 1
            domain_sid_val[domain][task]["total"] += 1

            res = match_sid(task,tl,best,sid_map)
            if res==1:
                top1_stats[task]["right"]+=1
                sid_val[task]["val"]+=1

                domain_top1_stats[domain][task]["right"] += 1
                domain_sid_val[domain][task]["val"] += 1

            elif res==0:
                sid_val[task]["val"]+=1
                domain_sid_val[domain][task]["val"] += 1
        else:
            top1_stats[task]["total"]+=1
            domain_top1_stats[domain][task]["total"] += 1
            if match_top1(label,best):
                top1_stats[task]["right"]+=1
                domain_top1_stats[domain][task]["right"] += 1

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx+1} samples")

def acc(r,t): return (r/t)*100 if t else 0

print("=== SID-to-xx Alignment Report ===")
# for k in ["sid_to_name", "sid_to_domain", "sid_to_cp", "sid_to_year", "sid_to_paytype", "sid_to_ip", "sid_to_artist"]:
for k in ["sid_to_name", "sid_to_domain", "sid_to_cp", "sid_to_year", "sid_to_paytype", "sid_to_ip", "item_to_attr"]:
    r = top1_stats[k]["right"]
    t = top1_stats[k]["total"]
    print(f"{k}: acc={acc(r,t):.2f}% ({r}/{t})")
tr = tag_stats["right_cnt"]
tt = tag_stats["total_cnt"]
print(f"sid_to_tag: tag-level acc={acc(tr,tt):.2f}% (right_tags={tr} / total_tags={tt}), samples={tag_stats['sample_total']}")

print("=== xx-to-SID Alignment Report ===")
# for k in ["item_to_sid", "name_to_sid", "tag_to_sid", "artist_to_sid", "attr_to_sid", "ip_to_sid"]:
for k in ["item_to_sid", "name_to_sid", "tag_to_sid", "attr_to_sid", "ip_to_sid"]:
    r = top1_stats[k]["right"]
    t = top1_stats[k]["total"]
    print(f"{k}: acc={acc(r,t):.2f}% ({r}/{t})")

print("=== Domain-wise Alignment Report ===")
for domain in domain_top1_stats:
    print(f"\n--- Domain: {domain} ---")
    # for k in ["sid_to_name","sid_to_domain", "sid_to_cp", "sid_to_year", "sid_to_paytype", "sid_to_ip", "sid_to_artist","sid_to_artist",
    for k in ["sid_to_name","sid_to_domain", "sid_to_cp", "sid_to_year", "sid_to_paytype", "sid_to_ip", "sid_to_artist",
              "item_to_sid","name_to_sid","tag_to_sid","attr_to_sid", "ip_to_sid"]:
        s = domain_top1_stats[domain][k]
        if s["total"] > 0:
            print(f"{k}: acc={acc(s['right'],s['total']):.2f}% ({s['right']}/{s['total']})")

# print("=== Domain-wise SID Valid Report ===")
# for domain in domain_sid_val:
#     print(f"\n--- Domain: {domain} ---")
#     for k in ["item_to_sid","name_to_sid","tag_to_sid","artist_to_sid"]:
#         s = domain_sid_val[domain][k]
#         if s["total"] > 0:
#             print(f"{k}: val={acc(s['val'],s['total']):.2f}% ({s['val']}/{s['total']})")