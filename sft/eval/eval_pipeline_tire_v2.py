#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, json
import pandas as pd
from datetime import datetime, timezone
import time
from tqdm import tqdm


try:
    from trie import SIDTrie
    from constraint import (
        SIDConstrainedLogitsProcessor,
        CachedSIDConstrainedLogitsProcessor,
        FlatTrieProcessor,
        PrecomputedMaskProcessor,
        OptimizedBatchProcessor,
        HybridProcessor,
    )
except ImportError as e:
    logging.error(f"导入 trie / constraint 失败，请检查 PYTHONPATH: {e}")
# 配置：是否启用多版本对比模式
# True: 对比所有版本（用于评测）
# False: 只使用 Precomputed 优化版本（用于生产）
ENABLE_BENCHMARK_MODE = False  # 默认使用生产模式

# 生产环境默认使用 Precomputed 优化版本
DEFAULT_PROCESSOR = "precomputed"
DEFAULT_PRECOMPUTE_LEVELS = 2

# dataname = "merged_emb.npy_rqopq_base_3rq_1opq2_bigemb"
# dataname = "station_video_4B_json_format.npy_rqopq_base"
dataname = "20260303_rqkmeans.patched"

# MODEL_DIR = "/lizhaoxuan1/checkpoints/qwen3_4b_sft_0208_sft_single_merged_emb.npy_rqopq_base_3rq_1opq2_bigemb_gpu8_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"
MODEL_DIR = "/mnt/lizhaoxuan1/checkpoints/qwen3_4b_Instruct_sft_0305_sft_single_video_ext_20260303_rqkmeans.patched_gpu16_pktrue_ntpktrue_full_1024_8_bs32_ga2_1e-5/checkpoint-3000/"

SID_TRIE_PATH = f"/mnt/lizhaoxuan1/sid/result/sid_trie_{dataname}.pkl"
SID_PATH = f"/mnt/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"

EVAL_CSV_PATH = "/mnt/zhanggehang1/Sid/evalFlow/station_top_3000_query_0225.csv"
OUTPUT_CSV_PATH = "/mnt/yingchen1/evalFlow/submission_top_3000_results_0306_trie.jsonl"
OUTPUT_VAL_CSV_PATH = "/mnt/yingchen1/evalFlow/submission_top_3000_results_trie_llm_judge.jsonl"

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


def build_prompt(query, domain, gender, session, date="20260310"):
    return (
        "你的任务是基于“用户历史对话”和“用户的当前查询需求”，分析用户搜索意图，给出符合用户要求的资源的SID。\n\n"
        "### 用户的当前查询需求(可能包含多音字、识别错误或发音不准的情况。)\n"
        f"{query}\n\n"
        "### 用户请求信息\n"
        f"- 请求资源类型: {domain}\n"
        f"- 请求当前日期: {date}\n"
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

    # 从 requestfeature 中尝试解析 deviceId 
    device_id = ""
    try:
        req_feat = json.loads(row.get("requestfeature", "{}"))
        device_id = req_feat.get("deviceId", "")
    except Exception:
        pass

    return {
        "instruction": build_prompt(
            query,
            row.get("type", "电台"), # 使用实际类别，如果没有则默认电台
            "未知",
            "无"
        ),
        "request_id": row.get("requestid", ""),
        "device_id": device_id,
        "query": query,
        "name": row.get("name", ""),
        "type": row.get("type", "")
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
                    sid = "".join(d.get("indices", ""))
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

    # 根据模式初始化 processor
    if ENABLE_BENCHMARK_MODE:
        # 对比模式：初始化所有版本
        logger.info("=" * 60)
        logger.info("【对比模式】初始化所有优化版本...")
        logger.info("=" * 60)

        processors = {
            "original": SIDConstrainedLogitsProcessor(
                trie=sid_trie,
                tokenizer=tokenizer,
                force_constraint=True,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=None
            ),
            "cached": CachedSIDConstrainedLogitsProcessor(
                trie=sid_trie,
                tokenizer=tokenizer,
                force_constraint=True,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=None
            ),
            "flat": FlatTrieProcessor(
                trie=sid_trie,
                tokenizer=tokenizer,
                force_constraint=True,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=None
            ),
            "precomputed": PrecomputedMaskProcessor(
                trie=sid_trie,
                tokenizer=tokenizer,
                force_constraint=True,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=None,
                precompute_levels=DEFAULT_PRECOMPUTE_LEVELS
            ),
            "batch": OptimizedBatchProcessor(
                trie=sid_trie,
                tokenizer=tokenizer,
                force_constraint=True,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=None
            ),
            "hybrid": HybridProcessor(
                trie=sid_trie,
                tokenizer=tokenizer,
                force_constraint=True,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=None,
                precompute_levels=DEFAULT_PRECOMPUTE_LEVELS
            ),
        }

        # 统计信息结构
        processor_stats = {
            name: {
                "total_time": 0.0,
                "sample_count": 0,
                "total_generated": 0,
                "valid_count": 0,
                "times_window": [],
            }
            for name in processors.keys()
        }
        processor_stats["no_trie"] = {
            "total_time": 0.0,
            "sample_count": 0,
            "total_generated": 0,
            "valid_count": 0,
            "times_window": [],
        }
        main_processor = processors[DEFAULT_PROCESSOR]  # 用于获取结果
    else:
        # 生产模式：只使用 PrecomputedMaskProcessor
        logger.info("=" * 60)
        logger.info(f"【生产模式】使用 {DEFAULT_PROCESSOR} 优化版本...")
        logger.info("=" * 60)

        processors = {
            DEFAULT_PROCESSOR: PrecomputedMaskProcessor(
                trie=sid_trie,
                tokenizer=tokenizer,
                force_constraint=True,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=None,
                precompute_levels=DEFAULT_PRECOMPUTE_LEVELS
            )
        }
        main_processor = processors[DEFAULT_PROCESSOR]

        # 简化统计结构
        processor_stats = {
            DEFAULT_PROCESSOR: {
                "total_time": 0.0,
                "sample_count": 0,
                "total_generated": 0,
                "valid_count": 0,
                "times_window": [],
            },
            "no_trie": {
                "total_time": 0.0,
                "sample_count": 0,
                "total_generated": 0,
                "valid_count": 0,
                "times_window": [],
            }
        }

    df = pd.read_csv(EVAL_CSV_PATH)

    evals = []
    for _, row in df.iterrows():
        s = build_eval_sample(row)
        if s:
            evals.append(s)

    total_samples = len(evals)
    logger.info(f"Total eval samples: {total_samples}")

    result = []
    val_result = []

    # 用于计算实时速度的滑动窗口
    window_size = 10

    pbar = tqdm(enumerate(evals), total=total_samples, desc="推理进度", ncols=120,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for idx, item in pbar:
        messages = [{"role": "user", "content": item["instruction"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # 测试每个 processor（根据模式决定）
        if ENABLE_BENCHMARK_MODE:
            # 对比模式：前 5 个样本对比所有版本，后续只测关键版本
            if idx < 5:
                processors_to_test = list(processors.items())
            else:
                processors_to_test = [
                    ("original", processors["original"]),
                    ("hybrid", processors["hybrid"]),
                ]
        else:
            # 生产模式：只使用默认 processor
            processors_to_test = [(DEFAULT_PROCESSOR, main_processor)]

        preds_for_result = None

        for name, processor in processors_to_test:
            # 清空缓存（如果是缓存版本）
            if hasattr(processor, 'clear_cache'):
                processor.clear_cache()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=NUM_RETURN,
                    num_return_sequences=NUM_RETURN,
                    early_stopping=True,
                    logits_processor=[processor],
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()

            inference_time = end_time - start_time
            processor_stats[name]["total_time"] += inference_time
            processor_stats[name]["sample_count"] += 1
            processor_stats[name]["times_window"].append(inference_time)
            if len(processor_stats[name]["times_window"]) > window_size:
                processor_stats[name]["times_window"].pop(0)

            # 统计 SID 有效率
            preds = []
            for seq in outputs.sequences:
                gen_ids = seq[input_len:]
                sid_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                tokens = sid_trie.sid_str_to_tokens(sid_text)
                processor_stats[name]["total_generated"] += 1
                if sid_trie.search(tokens):
                    preds.append(sid_text)
                    processor_stats[name]["valid_count"] += 1

            # 保存 main_processor 的结果用于后续输出
            if processor == main_processor:
                preds_for_result = list(set(preds))

        # 无 Trie 约束的推理
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        no_trie_start = time.perf_counter()

        with torch.no_grad():
            no_trie_outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=NUM_RETURN,
                num_return_sequences=NUM_RETURN,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        no_trie_end = time.perf_counter()

        no_trie_time = no_trie_end - no_trie_start
        processor_stats["no_trie"]["total_time"] += no_trie_time
        processor_stats["no_trie"]["sample_count"] += 1
        processor_stats["no_trie"]["times_window"].append(no_trie_time)
        if len(processor_stats["no_trie"]["times_window"]) > window_size:
            processor_stats["no_trie"]["times_window"].pop(0)

        # 统计无约束 SID 有效率
        for seq in no_trie_outputs.sequences:
            gen_ids = seq[input_len:]
            sid_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            tokens = sid_trie.sid_str_to_tokens(sid_text)
            processor_stats["no_trie"]["total_generated"] += 1
            if sid_trie.search(tokens):
                processor_stats["no_trie"]["valid_count"] += 1

        # 更新进度条（显示关键指标）
        if ENABLE_BENCHMARK_MODE:
            # 对比模式：显示原始版本和优化版本
            if "original" in processor_stats and processor_stats["original"]["times_window"]:
                orig_avg = sum(processor_stats["original"]["times_window"]) / len(processor_stats["original"]["times_window"]) * 1000
                no_trie_avg = sum(processor_stats["no_trie"]["times_window"]) / len(processor_stats["no_trie"]["times_window"]) * 1000
                if "hybrid" in processor_stats and processor_stats["hybrid"]["times_window"]:
                    hybrid_avg = sum(processor_stats["hybrid"]["times_window"]) / len(processor_stats["hybrid"]["times_window"]) * 1000
                    pbar.set_postfix({
                        'Orig': f"{orig_avg:.0f}ms",
                        'Hybrid': f"{hybrid_avg:.0f}ms",
                        'NoTrie': f"{no_trie_avg:.0f}ms",
                    })
                else:
                    pbar.set_postfix({
                        'Orig': f"{orig_avg:.0f}ms",
                        'NoTrie': f"{no_trie_avg:.0f}ms",
                    })
        else:
            # 生产模式：只显示当前版本
            if DEFAULT_PROCESSOR in processor_stats and processor_stats[DEFAULT_PROCESSOR]["times_window"]:
                proc_avg = sum(processor_stats[DEFAULT_PROCESSOR]["times_window"]) / len(processor_stats[DEFAULT_PROCESSOR]["times_window"]) * 1000
                no_trie_avg = sum(processor_stats["no_trie"]["times_window"]) / len(processor_stats["no_trie"]["times_window"]) * 1000
                pbar.set_postfix({
                    'Precomp': f"{proc_avg:.0f}ms",
                    'NoTrie': f"{no_trie_avg:.0f}ms",
                })

        # 保存结果（使用 original 的结果）
        if preds_for_result is None:
            preds_for_result = []

        pred_source_name = []
        matchlevel = []
        for pred in preds_for_result:
            meta_info = sid_map.get(pred)
            if meta_info:
                pred_source_name.append(meta_info)
                matchlevel.append("ACCURATE")

        result.append({
            "request_id": item.get("request_id", ""),
            "device_id": item.get("device_id", ""),
            "query": item.get("query", ""),
            "name": item.get("name", ""),
            "type": item.get("type", ""),
            "matchlevel": matchlevel,
            "source_name": pred_source_name,
            "predict": preds_for_result,
        })

        val_result.append({
            "query": item.get("query", ""),
            "matchlevel": matchlevel,
            "source_name": pred_source_name,
        })

    # ========== 输出推理速度统计 ==========
    logger.info("\n" + "=" * 80)
    if ENABLE_BENCHMARK_MODE:
        logger.info("Trie 优化对比评估报告")
        logger.info("=" * 80)
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"前 5 个样本测试所有版本，后续样本测试关键版本")
        logger.info("")
    else:
        logger.info("推理性能报告")
        logger.info("=" * 80)
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"使用优化版本: {DEFAULT_PROCESSOR}")
        logger.info("")

    # 计算各项指标
    for name, stats in processor_stats.items():
        if stats["sample_count"] > 0:
            avg_time = stats["total_time"] / stats["sample_count"] * 1000  # ms
            throughput = stats["sample_count"] / stats["total_time"] if stats["total_time"] > 0 else 0
            valid_rate = stats["valid_count"] / stats["total_generated"] * 100 if stats["total_generated"] > 0 else 0

            stats["avg_ms"] = avg_time
            stats["throughput"] = throughput
            stats["valid_rate"] = valid_rate

    # 打印对比表格（仅对比模式）
    if ENABLE_BENCHMARK_MODE:
        logger.info("【性能对比表】")
        logger.info("-" * 80)
        logger.info(f"{'Version':<15} {'Avg Time':<12} {'Throughput':<12} {'Valid Rate':<12} {'Samples':<10}")
        logger.info("-" * 80)

        version_names = {
            "no_trie": "No Trie",
            "original": "Original",
            "cached": "Cached",
            "flat": "Flat Trie",
            "precomputed": "Precomputed",
            "batch": "Batch",
            "hybrid": "Hybrid",
        }

        if "original" in processor_stats:
            baseline_avg = processor_stats["original"].get("avg_ms", 1)
        else:
            baseline_avg = processor_stats.get(DEFAULT_PROCESSOR, {}).get("avg_ms", 1)

        for name in ["no_trie", "original", "cached", "flat", "precomputed", "batch", "hybrid"]:
            if name in processor_stats and processor_stats[name]["sample_count"] > 0:
                stats = processor_stats[name]
                display_name = version_names.get(name, name)
                speedup = baseline_avg / stats["avg_ms"] if stats["avg_ms"] > 0 else 1.0
                logger.info(f"{display_name:<15} {stats['avg_ms']:<12.2f} {stats['throughput']:<12.2f} "
                           f"{stats['valid_rate']:<12.1f} {stats['sample_count']:<10} "
                           f"({'+' if speedup > 1 else ''}{speedup:.2f}x)")

        logger.info("-" * 80)

        # 详细分析
        logger.info("")
        logger.info("【优化效果分析】")

        orig_avg = processor_stats["original"].get("avg_ms", 0) if "original" in processor_stats else 0
        no_trie_avg = processor_stats["no_trie"].get("avg_ms", 1)

        if orig_avg > 0:
            logger.info(f"1. 基准对比（vs 无约束）:")
            logger.info(f"   Original / No Trie = {orig_avg/no_trie_avg:.2f}x 慢")
            logger.info("")

            logger.info(f"2. 各优化版本相对 Original 的加速比:")
            for name in ["cached", "flat", "precomputed", "batch", "hybrid"]:
                if name in processor_stats and processor_stats[name]["sample_count"] > 0:
                    opt_avg = processor_stats[name]["avg_ms"]
                    speedup = orig_avg / opt_avg
                    improvement = (orig_avg - opt_avg) / orig_avg * 100
                    logger.info(f"   {version_names[name]:<12}: {speedup:.2f}x 快 ({improvement:+.1f}%)")

        logger.info("")

        # 额外统计信息
        for name, processor in processors.items():
            if hasattr(processor, 'get_cache_stats'):
                stats = processor.get_cache_stats()
                logger.info(f"【{version_names[name]} 缓存统计】")
                for key, value in stats.items():
                    if isinstance(value, float):
                        logger.info(f"   {key}: {value:.4f}")
                    else:
                        logger.info(f"   {key}: {value}")
                logger.info("")

            if hasattr(processor, 'get_precompute_stats'):
                stats = processor.get_precompute_stats()
                logger.info(f"【{version_names[name]} 预计算统计】")
                for key, value in stats.items():
                    if isinstance(value, float):
                        logger.info(f"   {key}: {value:.4f}")
                    else:
                        logger.info(f"   {key}: {value}")
                logger.info("")
    else:
        # 生产模式：简化输出
        logger.info("【性能摘要】")
        logger.info("-" * 80)
        stats = processor_stats[DEFAULT_PROCESSOR]
        logger.info(f"处理器: {DEFAULT_PROCESSOR}")
        logger.info(f"平均耗时: {stats['avg_ms']:.2f} ms")
        logger.info(f"吞吐量: {stats['throughput']:.2f} samples/sec")
        logger.info(f"SID 有效率: {stats['valid_rate']:.1f}%")
        logger.info(f"总样本: {stats['sample_count']}")

        # 显示无约束对比
        no_trie_stats = processor_stats["no_trie"]
        if no_trie_stats["sample_count"] > 0:
            no_trie_avg = no_trie_stats["avg_ms"]
            speedup = no_trie_avg / stats["avg_ms"]
            logger.info(f"相对无约束加速: {speedup:.2f}x")
        logger.info("-" * 80)

        # 预计算统计
        if hasattr(main_processor, 'get_precompute_stats'):
            precompute_stats = main_processor.get_precompute_stats()
            logger.info("")
            logger.info("【预计算统计】")
            for key, value in precompute_stats.items():
                if isinstance(value, float):
                    logger.info(f"   {key}: {value:.4f}")
                else:
                    logger.info(f"   {key}: {value}")

    logger.info("=" * 80)

    logger.info("=" * 80)

    result_df = pd.DataFrame(result)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False)

    val_result_df = pd.DataFrame(val_result)
    val_result_df.to_csv(OUTPUT_VAL_CSV_PATH, index=False)

    logger.info(f"Done -> {OUTPUT_VAL_CSV_PATH}")


if __name__ == "__main__":
    main()
