#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== 你已有的模块 =====================
from trie import SIDTrie
from constraint import SIDConstrainedLogitsProcessor

# ================== 路径配置 ==================
MODEL_DIR = "/lizhaoxuan1/checkpoints/qwen3_4b_sft_0205_align_train_station_video_4B_json_format.npy_rqopq_base.index_station_0.95_video_0.98_station_video_4B_json_format.npy_rqopq_base_gpu8_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/checkpoint-800/"

INPUT_PATH = "/lizhaoxuan1/train_data/align_train_station_video_4B_json_format.npy_rqopq_base.index_station_0.95_video_0.98_station_video_4B_json_format.npy_rqopq_base.index.jsonl"

OUTPUT_PATH = (
    "/lizhaoxuan1/llm_output/"
    "align_infer_constrained_train.jsonl"
)

SID_TRIE_PATH = "/lizhaoxuan1/sid/result/sid_trie.pkl"
SID_MAPPING_PATH = "/lizhaoxuan1/sid/result/station_video_4B_json_format.npy_rqopq_base.index.jsonl"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

MAX_INFER_NUM = 200
# ================== 推理参数 ==================
MAX_NEW_TOKENS = 16
NUM_RETURN = 5
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1

# 只有这些任务才启用 SID 约束
SID_TASKS = {
    "item_to_sid",
    "name_to_sid",
    "tag_to_sid",
    "artist_to_sid",
}

# 不使用约束
# SID_TASKS = {
# }


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================
# Step 0. 构建 / 加载 SID Trie
# ======================================================
def build_trie_from_sid_data(sid_mapping_file: str) -> SIDTrie:
    sids = set()
    pattern = r"<a_(\d+)><b_(\d+)><c_(\d+)><d_(\d+)><e_(\d+)>"

    with open(sid_mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                sid = "".join(rec.get("indices", ""))
                m = re.match(pattern, sid)
                if m:
                    sids.add(tuple(int(x) for x in m.groups()))
            except Exception:
                continue

    logger.info(f"找到 {len(sids)} 个 SID")

    max_vals = [0] * 5
    for sid in sids:
        for i, v in enumerate(sid):
            max_vals[i] = max(max_vals[i], v)

    codebook_sizes = [v + 100 for v in max_vals]
    logger.info(f"Codebook sizes: {codebook_sizes}")

    trie = SIDTrie(num_levels=5, codebook_sizes=codebook_sizes)
    for sid in sids:
        trie.insert(list(sid))

    logger.info(f"Trie 构建完成，SID 数={trie.num_sids}")
    return trie


if os.path.exists(SID_TRIE_PATH):
    sid_trie = SIDTrie.load(SID_TRIE_PATH)
    logger.info(f"Loaded SID Trie from {SID_TRIE_PATH}")
else:
    sid_trie = build_trie_from_sid_data(SID_MAPPING_PATH)
    sid_trie.save(SID_TRIE_PATH)
    logger.info(f"Saved SID Trie to {SID_TRIE_PATH}")

# ======================================================
# Step 1. 加载模型 & tokenizer
# ======================================================
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

# ======================================================
# Step 2. 构建 SID 约束 LogitsProcessor
# ======================================================
sid_logits_processor = SIDConstrainedLogitsProcessor(
    trie=sid_trie,
    tokenizer=tokenizer,
    force_constraint=True,
    eos_token_id=eos_id,
    sep_token_id=None,
)

# ======================================================
# 工具函数
# ======================================================
def drop_candidate_sid_list(text: str) -> str:
    return re.sub(r"从候选SID列表.*?中[,，]", "从已学习过的有效SID中", text, flags=re.DOTALL)


def build_inputs(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt").to(model.device)


def decode_one(seq, prompt_len: int) -> str:
    return tokenizer.decode(
        seq[prompt_len:],
        skip_special_tokens=True
    ).strip()


# ======================================================
# 推理主循环
# ======================================================
with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

    for idx, line in enumerate(f_in):
        try:
            data = json.loads(line)
        except Exception:
            continue

        task_type = data.get("task_type")
        messages = data.get("messages", [])
        if not messages or len(messages) < 2:
            continue

        prompt = drop_candidate_sid_list(messages[0]["content"].strip())
        label = messages[1]["content"].strip()

        inputs = build_inputs(prompt)
        prompt_len = inputs["input_ids"].shape[1]

        use_constraint = task_type in SID_TASKS
        logits_processors = [sid_logits_processor] if use_constraint else None

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_beams=1,
                num_return_sequences=NUM_RETURN,
                eos_token_id=eos_ids,
                pad_token_id=eos_id,
                repetition_penalty=REPETITION_PENALTY,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=logits_processors,
            )


        sequences = out.sequences
        seq_scores = None
        if not DO_SAMPLE:
            seq_scores = out.sequences_scores  # shape: [num_return_sequences]

        predicts = []
        scores = []
        for i, seq in enumerate(sequences):
            predicts.append(decode_one(seq, prompt_len))
            scores.append(float(seq_scores[i].item()) if seq_scores is not None else None)

        # beam 下 sequences_scores 越大越好（通常是log-prob/length penalty后的值）
        best_idx = max(range(len(predicts)), key=lambda i: scores[i] if scores[i] is not None else -1e30)
        best_predict = predicts[best_idx]
        best_score = scores[best_idx]

        out_obj = {
            "task_type": task_type,
            "idx": idx,
            "prompt": prompt,
            "label": label,
            "predicts": predicts,
            "scores": scores,
            "top1_predict": best_predict,
            "top1_score": best_score
        }
        f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx+1} samples")

        if idx > MAX_INFER_NUM:
            break

logger.info(f"Inference finished. Output -> {OUTPUT_PATH}")
