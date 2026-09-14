#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import logging
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ================== 路径配置 ==================
dataname = "station_video_4B_json_format.npy_rqopq_base.index"

model_dir = "/lizhaoxuan1/checkpoints/qwen3_4b_sft_0130_sft_single_station_video_4B_json_format.npy_rqopq_base_gpu8_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"
input_path = f"/lizhaoxuan1/train_data/sft_single_{dataname}_eval.jsonl"
output_path = f"/lizhaoxuan1/llm_output/sft_single_{dataname}_infer_beam.jsonl"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def remove_think_tag(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def main():
    parser = argparse.ArgumentParser(description="HF Beam Search inference on JSONL")
    parser.add_argument("--model_path", type=str, default=model_dir)
    parser.add_argument("--input_jsonl", type=str, default=input_path)
    parser.add_argument("--output_jsonl", type=str, default=output_path)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    args = parser.parse_args()

    # 创建输出目录
    output_file = Path(args.output_jsonl)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    eos_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [x for x in [eos_id, im_end_id] if x is not None]

    # ================= 读取输入 =================
    prompts = []
    original_lines = []

    logger.info(f"Reading input from: {args.input_jsonl}")
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            if "input" not in data:
                continue

            user_input = data["instruction"]

            if isinstance(user_input, str):
                messages = [{"role": "user", "content": user_input}]
            elif isinstance(user_input, list):
                messages = user_input
            else:
                continue

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            prompts.append(prompt)
            original_lines.append(data)

    logger.info(f"Loaded {len(prompts)} samples.")

    # ================= Beam Search 推理 =================
    logger.info(
        f"Starting HF Beam Search: num_beams={args.num_beams}, "
        f"num_return_sequences={args.num_beams}"
    )

    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                max_new_tokens=args.max_tokens,
                eos_token_id=eos_ids,
                pad_token_id=eos_id,
                early_stopping=True,
                repetition_penalty=1.1,
            )

        preds = []
        for seq in outputs:
            gen_ids = seq[inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            text = remove_think_tag(text)
            preds.append(text)

        # 和 sampling 版保持一致：去重 + ';' 拼接
        results.append(";".join(dict.fromkeys(preds)))

    # ================= 写出结果 =================
    logger.info(f"Writing results to: {args.output_jsonl}")
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for orig, beam_pred in zip(original_lines, results):
            orig["predict"] = beam_pred
            fout.write(json.dumps(orig, ensure_ascii=False) + "\n")

    logger.info("✅ HF Beam Search inference completed!")


if __name__ == "__main__":
    main()
