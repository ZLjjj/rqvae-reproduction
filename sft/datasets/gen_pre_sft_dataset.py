import ast
import json
from typing import Dict, Any, List
import random
from collections import defaultdict, Counter

# ========= split config =========
dataname = "vs_Qwen4B_rqvae_hc_ema"
SID_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"

TRAIN_FILE_NAME = f"pre_sft_video_{dataname}"
OUTPUT_TRAIN_STAFF_JSONL_PATH = f"/lizhaoxuan1/train_data/{TRAIN_FILE_NAME}.jsonl"
DATASET_INFO_PATH = "/lizhaoxuan1/gensearchrec/llama-factory/data/dataset_info.json"

BELLA_PATH = "/lizhaoxuan1/raw_data/Belle_2M_CN.json"
ALBACA_PATH = "/lizhaoxuan1/raw_data/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"

random.seed(2026)

director_black_list = ["空中英语", "空中英语教室", "佚名", "暂无", "无"]
actor_black_list = ["佚名", "深圳悦道", "-", "读书部", "读书郎", "九学王", "智硕科技", "新学未", "顺飒-教育VIP", "空中英语"]

STAFF_TASK_NUM = 10
STAFF_IP_NUM = 5
TOP_ACTOR_NUM = 2000
TOP_DIRECTOR_NUM = 500
MIN_IP_SID_NUM = 2

# ================= SID 构建 =================
def build_sid_from_indices(indices):
    if isinstance(indices, list):
        cleaned = [str(x).strip() for x in indices if x not in (None, "", " ")]
        if cleaned:
            return "".join(cleaned)

    if isinstance(indices, str):
        try:
            arr = ast.literal_eval(indices)
            if isinstance(arr, list):
                cleaned = [str(x).strip() for x in arr if x not in (None, "", " ")]
                return "".join(cleaned)
        except Exception:
            return indices.strip()

    return "UNKNOWN_SID"

# ================= Prompt 构建 =================
def build_prompt(query, domain, ymd, asr_gender, formatted_session):
    return (
        "你的任务是基于“用户历史对话”和“用户的当前查询需求”，分析用户搜索意图，给出符合用户要求的资源的SID。\n\n"
        "### 用户的当前查询需求(可能包含多音字、识别错误或发音不准的情况。)\n"
        f"{query}\n\n"
        "### 用户请求信息\n"
        f"- 请求资源类型: {domain}\n"
        f"- 请求当前日期: {ymd}\n"
        f"- 语音识别性别: {asr_gender}\n\n"
        "### 用户历史对话\n"
        f"{formatted_session}\n\n"
        "### 筛选与匹配策略\n"
        "1. **用户意图识别**：结合用户历史对话信息，分析用户当前查询需求是否存在多音字、识别错误或用户发音不准的情况。\n"
        "2. **结果生成**：优先返回完全匹配用户查询意图的资源SID，若无完全匹配的资源，则返回最贴近用户需求的资源SID\n"
        "3. **完全匹配判定**：若候选资源满足“原始Query”或“修复Query”之一的全部需求，则判定为完全匹配。\n\n"
        "### 输出要求\n"
        "只需要资源SID，不需要输出任何具体理由。"
    )

# ================ 通用数据集 =====================
import pandas as pd
def load_parquet_records(file_path):
    df = pd.read_parquet(file_path)
    return df.to_dict(orient="records")

def gen_general_sft_data(max_alpaca=100000, max_coig=100000):
    tasks = []
    alpaca_tasks = []
    bella_tasks = []

    # ===== Alpaca =====
    alpaca_data = load_parquet_records(ALBACA_PATH)

    alpaca_sample = random.sample(
        alpaca_data,
        min(len(alpaca_data), max_alpaca)
    )

    for rec in alpaca_sample:
        instruction = rec.get("instruction", "")
        input_text = rec.get("input", "")
        output = rec.get("output", "")

        if not instruction or not output:
            continue

        user_text = instruction + ("\n" + input_text if input_text else "")

        alpaca_tasks.append({
            "instruction": user_text,
            "input": "",
            "output": output,
            "task_type": "general_en"
        })

    print(f"[DONE] Alpaca 样本: {len(alpaca_tasks)}")

    # ===== BELLE =====
    belle_data = []
    with open(BELLA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                belle_data.append(json.loads(line))
            except:
                continue

    belle_sample = random.sample(
        belle_data,
        min(len(belle_data), max_coig)
    )

    for rec in belle_sample:
        instruction = rec.get("instruction", "")
        input_text = rec.get("input", "")
        output = rec.get("output", "")

        if not instruction or not output:
            continue

        user_text = instruction + ("\n" + input_text if input_text else "")

        bella_tasks.append({
            "instruction": user_text,
            "input": "",
            "output": output,
            "task_type": "general_zh"
        })

    print(f"[DONE] BELLE 样本: {len(bella_tasks)}")

    # ===== 合并 =====
    tasks = bella_tasks + alpaca_tasks
    random.shuffle(tasks)

    return tasks

# ================= 统计视频元信息 =================
def cnt_video_meta_info():
    director_dict = defaultdict(list)
    actor_dict = defaultdict(list)
    ip_dict = defaultdict(list)
    name_dict = defaultdict(list)
    station_artist_dict = defaultdict(list)

    director_counter = Counter()
    actor_counter = Counter()
    station_artist_counter = Counter()

    with open(SID_PATH, "r", encoding="utf-8") as f_in:
        for line_num, line in enumerate(f_in):
            line = line.strip()
            if not line or line_num < 1100000:
                continue

            try:
                rec_raw = json.loads(line)
            except Exception:
                continue

            if rec_raw.get("domain") != "video":
                continue

            meta = rec_raw.get("meta")
            if not meta:
                continue
            
            if isinstance(meta, dict):
                meta_dict = meta
            elif isinstance(meta, str):
                try:
                    meta_dict = json.loads(meta)
                except Exception:
                    continue
            else:
                continue

            station_artist = meta_dict.get("br_name")
            ip_name = meta_dict.get("patchwall_ip_name")
            directors = meta_dict.get("directors") or []
            actors = meta_dict.get("main_actors") or []
            name = meta_dict.get("meta_origin_name")
            video_type = meta_dict.get("video_type")

            sid = build_sid_from_indices(rec_raw.get("sid") or rec_raw.get("indices"))

            if name and video_type in ["电视剧","电影","综艺","动漫","儿童","纪录片"]:
                import unicodedata
                name = "".join(
                    ch for ch in name
                    if not unicodedata.category(ch).startswith("P")
                    and not ch.isspace()
                ) 
                name_dict[name].append(sid)

            if not station_artist and not ip_name and not directors and not actors:
                continue

            # ===== IP =====
            if ip_name:
                ip_dict[ip_name].append(sid)

            # if station_artist:
            #     station_artist_dict[station_artist].append(sid)
            #     station_artist_counter[station_artist] += 1

            # ===== 导演 =====
            for director in directors:
                if director and director not in director_black_list:
                    director_dict[director].append(sid)
                    director_counter[director] += 1

            # ===== 演员 =====
            for actor in actors:
                if actor and actor not in actor_black_list:
                    actor_dict[actor].append(sid)
                    actor_counter[actor] += 1

    # 只保留高频导演/演员  
    # TODO 从日志挖掘槽位作为白名单补充召回
    top_directors = {name for name, _ in director_counter.most_common(TOP_DIRECTOR_NUM)}
    top_actors = {name for name, _ in actor_counter.most_common(TOP_ACTOR_NUM)}
    top_station_artist = {name for name, _ in station_artist_counter.most_common(TOP_ACTOR_NUM)}

    director_dict = {k: v for k, v in director_dict.items() if k in top_directors}
    actor_dict = {k: v for k, v in actor_dict.items() if k in top_actors}
    station_artist_dict = {k: v for k, v in station_artist_dict.items() if k in top_station_artist}

    return ip_dict, director_dict, actor_dict, name_dict

# ================= 样本构建 =================
def dump_sample(task_type, user_text, assistant_text):
    return {
        "instruction": build_prompt(user_text, "视频", "20260226", "unknown", "无"),
        "input": "",
        "output": assistant_text,
        "task_type": task_type,
    }

# ================= 生成 SFT 数据 =================
def gen_pre_sft_staff_data(ip_dict, director_dict, actor_dict):
    total_cnt = 0
    tasks = []

    # ========= IP 任务 =========
    for ip_name, ip_sids in ip_dict.items():
        if not ip_name or len(ip_sids) < MIN_IP_SID_NUM:
            continue

        sids = list(set(ip_sids))

        # IP → SID
        for sid in sids:
            user_text = f"{ip_name}系列"
            tasks.append(dump_sample("ip_to_sid", user_text, sid))
            total_cnt += 1

        # for sid in random.sample(sids, min(len(sids), STAFF_IP_NUM)):
        #     user_text = f"SID:{sid}属于哪个影视IP系列"
        #     tasks.append(dump_sample("sid_to_ip", user_text, ip_name))
        #     total_cnt += 1

    # ========= 导演任务 =========
    for director, director_sids in director_dict.items():
        sids = list(set(director_sids))

        for sid in sids:
            user_text = f"{director}导演的影片"
            tasks.append(dump_sample("director_to_sid", user_text, sid))
            total_cnt += 1

        # for sid in random.sample(sids, min(len(sids), STAFF_TASK_NUM)):
        #     user_text = f"SID:{sid}的导演是谁"
        #     tasks.append(dump_sample("sid_to_director", user_text, director))
        #     total_cnt += 1

    # ========= 演员任务 =========
    for actor, actor_sids in actor_dict.items():
        sids = list(set(actor_sids))

        for sid in sids:
            user_text = f"{actor}主演的影片"
            tasks.append(dump_sample("actor_to_sid", user_text, sid))
            total_cnt += 1

        # for sid in random.sample(sids, min(len(sids), STAFF_TASK_NUM)):
        #     user_text = f"SID:{sid}的主演是谁"
        #     tasks.append(dump_sample("sid_to_actor", user_text, actor))
        #     total_cnt += 1

    # random.shuffle(tasks)
    print(f"[DONE] 生成ext_attr训练样本总数: {total_cnt}")
    return tasks

def gen_pre_sft_name_data(name_dict):
    total_cnt = 0
    tasks = []

    # ========= name 任务 =========
    for name, sids in name_dict.items():
        if not name or len(sids) < 1:
            continue

        sids = list(set(sids))

        # NAME → SID
        for sid in sids:
            user_text = f"{name}"
            tasks.append(dump_sample("name_to_sid", user_text, sid))
            total_cnt += 1

    # random.shuffle(tasks)
    print(f"[DONE] 生成name训练样本总数: {total_cnt}")
    return tasks

# ================= 主函数 =================
def gen_pre_sft_trained_jsonl():
    ip_dict, director_dict, actor_dict, name_dict = cnt_video_meta_info()

    # ===== 业务数据 =====
    staff_tasks = gen_pre_sft_staff_data(ip_dict, director_dict, actor_dict)
    name_tasks = gen_pre_sft_name_data(name_dict)

    # ===== 通用能力数据 =====
    general_tasks = gen_general_sft_data(
        max_alpaca=100000,
        max_coig=100000
    )

    # ===== 融合 =====
    final_tasks = staff_tasks + name_tasks + general_tasks

    random.shuffle(final_tasks)
    print(f"[TOTAL] 总样本数: {len(final_tasks)}")

    with open(OUTPUT_TRAIN_STAFF_JSONL_PATH, "w", encoding="utf-8") as f_out:
        for rec in final_tasks:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"训练数据已保存: {OUTPUT_TRAIN_STAFF_JSONL_PATH}")

def update_dataset_info(key, file_path, config_path):
    item = {
        "file_name": file_path,
        "columns": {
        "prompt": "instruction",
        "response": "output"
        }
    }

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if key in data:
        print(f"[INFO] Key '{key}' already exists, will be overwritten.")
    else:
        print(f"[INFO] Adding new key '{key}' to dataset_info.json")

    data[key] = item

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] dataset_info.json updated.")

if __name__ == "__main__":
    gen_pre_sft_trained_jsonl()
    # update_dataset_info(TRAIN_FILE_NAME, OUTPUT_TRAIN_STAFF_JSONL_PATH, DATASET_INFO_PATH)

