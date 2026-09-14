#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, json
import pandas as pd
from datetime import datetime, timezone

from trie import SIDTrie
from constraint import SIDConstrainedLogitsProcessor

# ==============================
# 配置区
# ==============================

# dataname = "merged_emb.npy_rqopq_base_3rq_1opq2_bigemb"
# dataname = "station_video_4B_json_format.npy_rqopq_base"
dataname = "vs_Qwen4B_rqvae_hc_ema"

# MODEL_DIR = "/lizhaoxuan1/checkpoints/qwen3_4b_sft_0208_sft_single_merged_emb.npy_rqopq_base_3rq_1opq2_bigemb_gpu8_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"
MODEL_DIR = "/lizhaoxuan1/checkpoints/qwen3_4b_Instruct_sft_0417_sft_single_video_ext_vs_Qwen4B_rqvae_hc_ema_gpu24_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"

SID_TRIE_PATH = f"/lizhaoxuan1/sid/result/sid_trie_{dataname}.pkl"
SID_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"

EVAL_CSV_PATH = "/lizhaoxuan1/raw_data/soundbox_auto_eval_baseline.csv"
OUTPUT_CSV_PATH = f"/lizhaoxuan1/mify_output/mify_video_llm_0429_beam_{dataname}_infer_result.csv"
OUTPUT_VAL_CSV_PATH = f"/lizhaoxuan1/mify_output/mify_video_llm_0429_beam_{dataname}_infer_result_eval.csv"

MAX_NEW_TOKENS = 16
NUM_RETURN = 10
TEMPERATURE = 0.7
TOP_P = 0.9

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
# 时间处理函数
# ==============================
def format_time_ago(ts_ms, now_ms):
    try:
        now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        delta = now - dt
        if delta.days > 0:
            return f"{delta.days}天前"
        if delta.seconds >= 3600:
            return f"{delta.seconds // 3600}小时前"
        if delta.seconds >= 60:
            return f"{delta.seconds // 60}分钟前"
        return f"{delta.seconds}秒前"
    except Exception:
        return "时间未知"


def format_session(session_history, srvts):
    if not session_history or not srvts:
        return "无"
    now = int(float(srvts))
    lines = []
    for i, turn in enumerate(session_history, 1):
        if "timestamp" not in turn:
            continue
        if now - turn["timestamp"] > 600_000:
            continue
        lines.append(
            f"第{i}轮({format_time_ago(turn['timestamp'], now)})："
            f"用户：“{turn.get('query','')}”，系统：“{turn.get('to_speak','')}”"
        )
    return "\n".join(lines) if lines else "无"


def format_date(srvts):
    try:
        return datetime.fromtimestamp(int(srvts) / 1000, tz=timezone.utc).strftime("%Y%m%d")
    except Exception:
        return "无"


def build_prompt(query, domain, ymd, gender, session):
    return (
        "你的任务是基于“用户历史对话”和“用户的当前查询需求”，分析用户搜索意图，给出符合用户要求的资源的SID。\n\n"
        "### 用户的当前查询需求(可能包含多音字、识别错误或发音不准的情况。)\n"
        f"{query}\n\n"
        "### 用户请求信息\n"
        f"- 请求资源类型: {domain}\n"
        f"- 请求当前日期: {ymd}\n"
        f"- 语音识别性别: {gender}\n\n"
        "### 用户历史对话\n"
        f"{session}\n\n"
        "### 筛选与匹配策略\n"
        "1. **用户意图识别**：- 结合用户历史对话信息，分析用户当前查询需求是否存在多音字、语音识别错误或用户发音不准的情况，若存在则先修复用户查询请求，并用“原始请求”和“修复请求”同等优先级分别完成后续资源筛选。\n"
        "2. **结果生成** ：- 优先返回完全匹配用户查询意图的资源SID，若无完全匹配的资源，则返回最贴近用户需求的资源SID\n"
        "3. **完全匹配判定** ：- 若候选资源列表中的条目满足“原始Query”或“修复Query”之一所包含的全部用户需求（包括资源名、别名、类别、语言、发行地区、是否付费、风格、发行日期、季数、出品方、演员、导演、角色等），则判定为完全匹配。\n\n"
        "### 输出要求\n"
        "只需要资源SID，不需要输出任何具体理由。"
    )


def build_eval_sample(row):
    query = row.get("query", "")
    if not query or pd.isna(query):
        return None
    try:
        history = json.loads(row.get("session_history", "[]"))
    except Exception:
        history = []

    return {
        "instruction": build_prompt(
            query,
            "视频",
            format_date(row.get("srvts")),
            row.get("asr_gender", "未知"),
            format_session(history, row.get("srvts"))
        ),
        "request_id": row.get("request_id", ""),
        "session_id": row.get("session_id", ""),
        "device_id": row.get("device_id", ""),
        "query": query,
        "to_speak": row.get("to_speak", ""),
        "domain": row.get("domain", ""),
        "func_name": row.get("func_name", ""),
        "copilot_code": row.get("copilot_code", ""),
        "source_name": row.get("source_name", ""),
        "review_score": row.get("打分dc review", ""),
    }


# ==============================
# Trie 构建
# ==============================
def load_sid_map(path):
    sid_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            sid = re.sub(r"[\[\]'，, ]", "", "".join(r.get("sid", "")))
            meta = r.get("text_format") or r.get("meta")
            # meta_dict = json.loads(meta)
            if sid and meta:
                sid_map[sid] = meta
    return sid_map

def load_trie():
    try:
        return SIDTrie.load(SID_TRIE_PATH)
    except:
        logger.info("Building Trie...")
        trie = SIDTrie(num_levels=5, codebook_sizes=[2000]*5)
        with open(SID_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    sid = "".join(d.get("sid", ""))
                    m = re.match(r'<a_(\d+)><b_(\d+)><c_(\d+)><d_(\d+)><e_(\d+)>', sid)
                    if m:
                        trie.insert([int(x) for x in m.groups()])
                except:
                    continue
        trie.save(SID_TRIE_PATH)
        return trie


# ==============================
# 主流程
# ==============================

def main():
    sid_map = load_sid_map(SID_PATH)
    sid_trie = load_trie()

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    sid_logits_processor = SIDConstrainedLogitsProcessor(
        trie=sid_trie,
        tokenizer=tokenizer,
        force_constraint=True,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=None
    )

    df = pd.read_csv(EVAL_CSV_PATH)

    evals = []
    for _, row in df.iterrows():
        s = build_eval_sample(row)
        if s:
            evals.append(s)

    logger.info(f"Total eval samples: {len(evals)}")

    result = []
    val_result = []

    for item in evals:
        messages = [{"role": "user", "content": item["instruction"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

# sampling search start
        # with torch.no_grad():
        #     outputs = model.generate(
        #         **inputs,
        #         max_new_tokens=MAX_NEW_TOKENS,
        #         do_sample=True,
        #         temperature=TEMPERATURE,
        #         top_p=TOP_P,
        #         num_return_sequences=NUM_RETURN,
        #         logits_processor=[sid_logits_processor],
        #         pad_token_id=tokenizer.eos_token_id,
        #     )

        # input_len = inputs["input_ids"].shape[1]

        # preds = []
        # for seq in outputs:
        #     sid_text = tokenizer.decode(
        #         seq[input_len:], skip_special_tokens=True
        #     ).strip()

        #     tokens = sid_trie.sid_str_to_tokens(sid_text)
        #     if sid_trie.search(tokens):
        #         preds.append(sid_text)

        # preds = list(set(preds))

# beam search start
        with torch.no_grad():
            beam_outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                  # 关闭采样
                num_beams=NUM_RETURN,             # beam 数 = 你原来想要的返回数
                num_return_sequences=NUM_RETURN,  # 返回 N 个结果
                early_stopping=True,
                logits_processor=[sid_logits_processor],
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        input_len = inputs["input_ids"].shape[1]

        preds = []

        for i, seq in enumerate(beam_outputs.sequences):
            gen_ids = seq[input_len:]

            sid_text = tokenizer.decode(
                gen_ids, skip_special_tokens=True
            ).strip()

            tokens = sid_trie.sid_str_to_tokens(sid_text)

            if sid_trie.search(tokens):
                preds.append(sid_text)

        preds = list(set(preds))
# beam search done

        pred_source_name = []
        matchlevel = []
        for pred in preds:
            meta_info = sid_map.get(pred)
            if meta_info:
                pred_source_name.append(meta_info)
                matchlevel.append("ACCURATE")

        result.append({
            "request_id": item["request_id"],
            "session_id": item["session_id"],
            "device_id": item["device_id"],
            "query": item["query"],
            "to_speak": item["to_speak"],
            "domain": item["domain"],
            "func_name": item["func_name"],
            "copilot_code": item["copilot_code"],
            "cp_list": "",
            "matchlevel": matchlevel,
            "source_name": pred_source_name,
            "origin_source_name": item["source_name"],
            "review_score": item["review_score"],
            "predict": preds,
        })

        val_result.append({
            "copilot_code": item["copilot_code"],
            "cp_list": "",
            "matchlevel": matchlevel,
            "query": item["query"],
            "source_name": pred_source_name,
            "to_speak": item["to_speak"],
        })

    result_df = pd.DataFrame(result)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False)

    val_result_df = pd.DataFrame(val_result)
    val_result_df.to_csv(OUTPUT_VAL_CSV_PATH, index=False)

    logger.info(f"Done -> {OUTPUT_VAL_CSV_PATH}")


if __name__ == "__main__":
    main()
