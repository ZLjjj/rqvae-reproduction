import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

model_dir = "/mnt/lizhaoxuan1/checkpoints/qwen3_4b_Instruct_sft_0219_sft_single_rqvae_opq_20260208_station_video_text_format_4B_gpu24_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

messages = [
    {
        "role": "user",
        "content": "你的任务是基于“用户历史对话”和“用户的当前查询需求”，分析用户搜索意图，给出符合用户要求的资源的SID。\n\n### 用户的当前查询需求(可能包含多音字、识别错误或发音不准的情况。)\n播放双女主虐恋故事\n\n### 用户请求信息\n- 请求资源类型: 电台\n- 请求当前日期: 无\n- 语音识别性别: 未知\n\n### 用户历史对话\n无\n\n### 筛选与匹配策略\n1. **用户意图识别**：- 结合用户历史对话信息，分析用户当前查询需求是否存在多音字、语音识别错误或用户发音不准的情况，若存在则先修复用户查询请求，并用“原始请求”和“修复请求”同等优先级分别完成后续资源筛选。\n2. **结果生成** ：- 优先返回完全匹配用户查询意图的资源SID，若无完全匹配的资源，则返回最贴近用户需求的资源SID\n3. **完全匹配判定** ：- 若候选资源列表中的条目满足“原始Query”或“修复Query”之一所包含的全部用户需求（包括资源名、别名、类别、语言、发行地区、是否付费、风格、发行日期、季数、出品方、演员、导演、角色等），则判定为完全匹配。\n\n### 输出要求\n只需要资源SID，不需要输出任何具体理由。"
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
# 加载映射文件
# =====================================================
# 映射地址
index_path = "/mnt/lizhaoxuan1/sid/result/rqvae_opq_20260208_station_video_text_format_4B.index.meta.jsonl"
print(f"Loading index map from {index_path}...")

sid_map = {}
try:
    with open(index_path, "r", encoding="utf-8") as f:
        # 使用 tqdm 显示加载进度，因为文件可能很大
        for line in tqdm(f):
            try:
                item = json.loads(line)
                # indices 是一个列表，如 ["<a_405>", "<b_243>", ...]
                # 将其拼接成字符串作为 key: "<a_405><b_243>..."
                key = "".join(item.get("indices", []))
                
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

# =====================================================
# 推理方式一：Beam Search
# =====================================================
# print("\n================ Beam Search (num_beams=5) ================\n")

# with torch.no_grad():
#     beam_outputs = model.generate(
#         **inputs,
#         max_new_tokens=16,
#         do_sample=False,
#         num_beams=5,
#         num_return_sequences=5,
#         early_stopping=True,
#         eos_token_id=eos_ids,
#         pad_token_id=eos_id,
#         repetition_penalty=1.1,
#         output_scores=True,
#         return_dict_in_generate=True
#     )

# for i, seq in enumerate(beam_outputs.sequences):
#     gen_ids = seq[inputs["input_ids"].shape[1]:]
#     ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
#     score = beam_outputs.sequences_scores[i].item()
#     print(f"[Beam {i}] score={ans} | {score:.4f}")

# =====================================================
# 推理方式二：Sampling
# =====================================================
print("\n================ Sampling (do_sample=True) ================\n")

with torch.no_grad():
    sample_outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=10,
        eos_token_id=eos_ids,
        pad_token_id=eos_id,
        repetition_penalty=1.1,
    )

print("-" * 50)
for i, seq in enumerate(sample_outputs):
    gen_ids = seq[inputs["input_ids"].shape[1]:]
    ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    # 在映射表中查找资源信息
    # 注意：生成的 ans 可能包含空格或其他分隔符，需要确保与 map 中的 key 格式一致
    # 假设 map key 是紧凑的 token 串 "<a_x><b_y>..."
    clean_ans = ans.replace(" ", "") 
    resource_info = sid_map.get(clean_ans, "未找到对应的资源信息")

    print(f"[Sample {i}] SID: {ans}")
    print(f"    -> Info: {resource_info}")