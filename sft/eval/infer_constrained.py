import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import re
import os
import pickle

from trie import SIDTrie
from constraint import SIDConstrainedLogitsProcessor

# =========================
# 配置
# =========================
MODEL_DIR = "/lizhaoxuan1/checkpoints/qwen3_4b_Instruct_sft_0415_align_train_station_0.95_video_0.99_vs_Qwen4B_rqvae_hc_ema_v2_gpu24_pktrue_ntpktrue_full_1024_5_bs32_ga2_1e-5/"
dataname = "vs_Qwen4B_rqvae_hc_ema"
INDEX_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"
SID_TRIE_PATH = f"/lizhaoxuan1/sid/result/sid_trie_{dataname}.pkl"

USE_CONSTRAINT = True
NEED_DETAILS = True

MAX_NEW_TOKENS = 16
NUM_BEAMS = 5
NUM_RETURN = 5

# =========================
# 加载模型
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# =========================
# SID map
# =========================
sid_map = {}
if NEED_DETAILS:
    print(f"Loading index map from {INDEX_PATH}...")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                item = json.loads(line)
                key = "".join(item.get("sid", []))
                value = item.get("text_format", "") or item.get("meta", "")
                if key:
                    sid_map[key] = value
            except:
                continue
    print(f"Loaded {len(sid_map)} items.")

# =========================
# Trie（和大脚本完全一致）
# =========================
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
                continue

    max_vals = [0]*5
    for sid in sids:
        for i, v in enumerate(sid):
            max_vals[i] = max(max_vals[i], v)

    trie = SIDTrie(num_levels=5, codebook_sizes=[v+100 for v in max_vals])
    for sid in sids:
        trie.insert(list(sid))

    return trie


if USE_CONSTRAINT:
    if os.path.exists(SID_TRIE_PATH):
        sid_trie = SIDTrie.load(SID_TRIE_PATH)
        print("[INFO] Trie loaded")
    else:
        sid_trie = build_trie_from_sid_data(INDEX_PATH)
        sid_trie.save(SID_TRIE_PATH)
        print("[INFO] Trie built & saved")

    eos_id = tokenizer.eos_token_id

    sid_logits_processor = SIDConstrainedLogitsProcessor(
        trie=sid_trie,
        tokenizer=tokenizer,
        force_constraint=True,
        eos_token_id=eos_id,
        sep_token_id=None,
    )
else:
    sid_logits_processor = None

# =========================
# 输入
# =========================
messages = [
    {"role": "user", "content": "2025年的柯南电影"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

eos_id = tokenizer.eos_token_id
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
eos_ids = [x for x in [eos_id, im_end_id] if x is not None]

# =========================
# 推理
# =========================
print("\n===== Inference =====\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_RETURN,
        early_stopping=True,
        eos_token_id=eos_ids,
        pad_token_id=eos_id,

        # ⭐ 关键：和训练脚本一致
        logits_processor=[sid_logits_processor] if USE_CONSTRAINT else None,

        output_scores=True,
        return_dict_in_generate=True
    )

# =========================
# 解码（关键修复点）
# =========================
def normalize_sid(text):
    # ✅ 严格抽取
    sid = "".join(re.findall(r"<[a-e]_\d+>", text))
    return sid.strip()

for i, seq in enumerate(outputs.sequences):
    gen_ids = seq[inputs["input_ids"].shape[1]:]

    raw_ans = tokenizer.decode(gen_ids, skip_special_tokens=True)
    ans = normalize_sid(raw_ans)

    score = outputs.sequences_scores[i].item()

    # ✅ 强校验（你之前缺这个）
    is_valid = ans in sid_map

    if NEED_DETAILS:
        resource_info = sid_map.get(ans, "❌ 未找到（非法SID）")

        print(f"[Beam {i}] SID: {ans} | score={score:.4f} | valid={is_valid}")
        print(f"    -> Info: {resource_info}")
    else:
        print(f"[Beam {i}] {ans} | score={score:.4f} | valid={is_valid}")