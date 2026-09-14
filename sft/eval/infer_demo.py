import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from itertools import islice
import json

MODEL_DIR = "/lizhaoxuan1/checkpoints/qwen3_4b_Instruct_sft_0417_sft_single_video_ext_vs_Qwen4B_rqvae_hc_ema_gpu24_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"
# MODEL_DIR = "/lizhaoxuan1/checkpoints/qwen3_4b_Instruct_sft_0304_align_train_station_0.95_video_0.98_20260303_rqkmeans.patched_gpu24_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/checkpoint-600/"

NEED_DETAILS = True
# NEED_DETAILS = False

# =====================================================
# 加载映射文件
# =====================================================
# 映射地址
index_path = "/lizhaoxuan1/sid/result/vs_Qwen4B_rqvae_hc_ema.index.meta.jsonl"
print(f"Loading index map from {index_path}...")

if NEED_DETAILS:
    sid_map = {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            # 使用 tqdm 显示加载进度，因为文件可能很大
            for line in tqdm(islice(f, 1_100_000, None)):
                try:
                    item = json.loads(line)
                    # indices 是一个列表，如 ["<a_405>", "<b_243>", ...]
                    # 将其拼接成字符串作为 key: "<a_405><b_243>..."
                    key = "".join(item.get("sid", []))
                    
                    # 获取资源信息，优先使用 text_format，如果没有则尝试从 meta 中提取
                    value = item.get("text_format", "")
                    if not value and "meta" in item:
                        value = item["meta"]
                    
                    if key:
                        sid_map[key] = value
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(sid_map)} items into map.")
    except FileNotFoundError:
        print(f"Error: Index file not found at {index_path}")
        sid_map = {}

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

messages = [
    {
        "role": "user",
        # "content": "请根据资源的文本信息:垂域：“视频”，原始标题：“第十一届中国金鹰电视艺术节”，主标题：“第十一届中国金鹰电视艺术节”，标签：“国产综艺、晚会、文化、文艺”，描述：“电视颁奖典礼、金鹰奖评选、明星阵容、行业论坛”，出版时间：“2016”，演员：“赵丽颖、霍建华、刘涛”，付费状态：“免费”。, 返回它对应的SID。"
        # "content": "赵丽颖的电视剧，输出一个SID"
        "content": "你的任务是基于“用户历史对话”和“用户的当前查询需求”，分析用户搜索意图，给出符合用户要求的资源的SID。\n\n### 用户的当前查询需求(可能包含多音字、识别错误或发音不准的情况。)\n播放凡人修仙传\n\n### 用户请求信息\n- 请求资源类型: 视频\n- 请求当前日期: 20260226\n- 语音识别性别: unknown\n\n### 用户历史对话\n无\n\n### 筛选与匹配策略\n1. **用户意图识别**：结合用户历史对话信息，分析用户当前查询需求是否存在多音字、识别错误或用户发音不准的情况。\n2. **结果生成**：优先返回完全匹配用户查询意图的资源SID，若无完全匹配的资源，则返回最贴近用户需求的资源SID\n3. **完全匹配判定**：若候选资源满足“原始Query”或“修复Query”之一的全部需求，则判定为完全匹配。\n\n### 输出要求\n只需要资源SID，不需要输出任何具体理由。"
        # "content": ""

    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

eos_id = tokenizer.eos_token_id
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
eos_ids = [x for x in [eos_id, im_end_id] if x is not None and x != tokenizer.unk_token_id]

# =====================================================
# 推理方式一：Beam Search
# =====================================================
print("\n================ Beam Search (num_beams=5) ================\n")

with torch.no_grad():
    beam_outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=True,
        eos_token_id=eos_ids,
        pad_token_id=eos_id,
        repetition_penalty=1,
        output_scores=True,
        return_dict_in_generate=True
    )

for i, seq in enumerate(beam_outputs.sequences):
    gen_ids = seq[inputs["input_ids"].shape[1]:]
    ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    score = beam_outputs.sequences_scores[i].item()
    # print(f"[Beam {i}] score={ans} | {score:.4f}")

    if NEED_DETAILS:
        clean_ans = ans.replace(" ", "") 
        resource_info = sid_map.get(clean_ans, "未找到对应的资源信息")

        print(f"[Beam {i}] SID: {ans}；score={ans} | {score:.4f}")
        print(f"    -> Info: {resource_info}")
    else:
        print(f"[Beam {i}] {ans}；score={ans} | {score:.4f}")

# =====================================================
# 推理方式二：Sampling
# =====================================================
# print("\n================ Sampling (do_sample=True) ================\n")

# with torch.no_grad():
#     sample_outputs = model.generate(
#         **inputs,
#         max_new_tokens=16,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         # top_k=20,
#         num_return_sequences=10,
#         eos_token_id=eos_ids,
#         pad_token_id=eos_id,
#         # repetition_penalty=1,
#     )

# for i, seq in enumerate(sample_outputs):
#     gen_ids = seq[inputs["input_ids"].shape[1]:]
#     ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
#     if NEED_DETAILS:
#         clean_ans = ans.replace(" ", "") 
#         resource_info = sid_map.get(clean_ans, "未找到对应的资源信息")

#         print(f"[Sample {i}] SID: {ans}")
#         print(f"    -> Info: {resource_info}")
#     else:
#         print(f"[Sample {i}] {ans}")
