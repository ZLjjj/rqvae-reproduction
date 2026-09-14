#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import logging
from pathlib import Path
from vllm import LLM, SamplingParams
import re

# ================== 路径配置 ==================
dataname = "rqvae_opq_20260208_station_video_text_format_4B"
index_jsonl_path = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"

model_dir = "/lizhaoxuan1/checkpoints/qwen3_4b_Base_sft_0224_sft_single_rqvae_opq_20260208_station_video_text_format_4B_video_only_gpu16_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"
input_path = f"/lizhaoxuan1/train_data/video_sft_single_val_rqvae_opq_20260208_station_video_text_format_4B.jsonl"
output_path = f"/lizhaoxuan1/llm_output/sft_single_{dataname}_infer_sampling.jsonl"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def remove_think_tag(text):
    # 非贪婪匹配，移除 <think> ... </think> 整个块（包括换行）
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

def main():
    parser = argparse.ArgumentParser(description="vLLM Qwen3 inference on JSONL with sampling n times per input")
    parser.add_argument("--model_path", type=str, default=model_dir, help="Path to Qwen3 checkpoint (local dir)")
    parser.add_argument("--input_jsonl", type=str,  default=input_path, help="Input JSONL file (with 'input' field as str or list of messages)")
    parser.add_argument("--output_jsonl", type=str,  default=output_path, help="Output JSONL file (with added 'predict' as list of strings)")
    parser.add_argument("--max_tokens", type=int, default=16, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (must be > 0 if num_samples > 1)")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate per input (n in vLLM SamplingParams)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="tensor_parallel_size")
    args = parser.parse_args()

    if args.num_samples > 1 and args.temperature <= 0.0:
        logger.warning("Temperature is 0.0 but num_samples > 1. Setting temperature=0.7 for diversity.")
        args.temperature = 0.7

    # 创建输出目录
    output_file = Path(args.output_jsonl)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # === 初始化 vLLM ===
    logger.info(f"Loading Qwen3 model from: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=args.num_samples,            
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
    include_stop_str_in_output=False
    )

    # === 逐行读取 + 构建 prompts（支持 chat template）===
    input_prompts = []
    original_lines = []

    logger.info(f"Reading input from: {args.input_jsonl}")
    with open(args.input_jsonl, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "input" not in data:
                logger.warning("Skipping line: missing 'input' field")
                continue

            user_input = data["instruction"]

            if isinstance(user_input, str):
                messages = [{"role": "user", "content": user_input}]
            elif isinstance(user_input, list):
                messages = user_input
            else:
                logger.warning(f"Unsupported 'input' type: {type(user_input)}. Skipping.")
                continue

            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.error(f"Failed to apply chat template to input: {user_input}. Error: {e}")
                continue

            input_prompts.append(prompt)
            original_lines.append(data)

    logger.info(f"Loaded {len(input_prompts)} samples for inference.")

    # 批量推理（vLLM 自动 batch + 并行）
    logger.info(f"Starting vLLM inference with n={args.num_samples} samples per input...")
    outputs = llm.generate(input_prompts, sampling_params)

    # 提取生成文本（每个输入对应 n 个输出）
    predictions_list = []
    for output in outputs:
        preds = []
        for candidate in output.outputs:  # output.outputs 是长度为 n 的列表
            pred_text = candidate.text.strip()
            pred_text = remove_think_tag(pred_text)
            preds.append(pred_text)
        predictions_list.append(preds)

    # 写入新 JSONL（保留原字段 + 新增 predict 为 list）
    logger.info(f"Writing results to: {args.output_jsonl}")
    with open(args.output_jsonl, 'w', encoding='utf-8') as fout:
        for orig, preds in zip(original_lines, predictions_list):
            orig["predict"] = ";".join(set(preds))  # 保持兼容：n=1 时直接字符串
            fout.write(json.dumps(orig, ensure_ascii=False) + '\n')

    logger.info("✅ Inference completed successfully!")

if __name__ == "__main__":
    main()