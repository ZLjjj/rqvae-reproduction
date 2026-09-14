#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import csv
import re
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

# --------------------------
# 通用配置
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
csv.field_size_limit(min(2**30, sys.maxsize))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# dataname = "merged_emb.npy_rqopq_base_3rq_1opq2_bigemb"
# dataname = "station_video_4B_json_format.npy_rqopq_base"
dataname = "vs_Qwen4B_rqvae_hc_ema"
SID_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"

VIDEO_TRAIN_PATH = "/lizhaoxuan1/train_data/train_dig.csv"
VIDEO_EVAL_PATH = "/lizhaoxuan1/train_data/eval_dig.csv"

# 视频的mock训练集，需要先运行 gen_pre_sft_dataset.py
VIDEO_EXT_ALIGN_PATH = f"/lizhaoxuan1/train_data/pre_sft_video_{dataname}.jsonl"

STATION_INPUT_PATH = "/lizhaoxuan1/train_data/station_cleaned_training_data.jsonl"
STATION_INPUT_PATH_EXT = "/lizhaoxuan1/train_data/test_sft_standard.jsonl"

STATION_EVAL_RATIO = 0.01

TRAIN_FILE_NAME = f"sft_single_video_ext_{dataname}"
VAL_FILE_NAME = f"sft_single_video_ext_val_{dataname}"
OUTPUT_TRAIN_JSONL_PATH = f"/lizhaoxuan1/train_data/{TRAIN_FILE_NAME}.jsonl"
VIDEO_OUTPUT_VAL_JSONL_PATH   = f"/lizhaoxuan1/train_data/video_{VAL_FILE_NAME}.jsonl"
STATION_OUTPUT_VAL_JSONL_PATH   = f"/lizhaoxuan1/train_data/station_{VAL_FILE_NAME}.jsonl"

DATASET_INFO_PATH = f"/lizhaoxuan1/gensearchrec/llama-factory/data/dataset_info.json"



def load_id_sid_map(index_file: str) -> dict:
    id_sid_map = {}
    sid_info_map = {}
    with open(index_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                id_val = record.get('id')
                sid_val = re.sub(r"[\[\]'，, ]", "", "".join(record.get("sid", "")))
                if id_val is not None and sid_val:
                    id_sid_map[str(id_val)] = str(sid_val)
                    sid_info_map[str(sid_val)] = record.get('text_format')
            except json.JSONDecodeError:
                continue
    logger.info(f"Loaded {len(id_sid_map)} ID→SID mappings.")
    return id_sid_map, sid_info_map


def format_time_ago(timestamp_ms: int, current_ts_ms: int) -> str:
    try:
        current = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        delta = current - dt
        if delta.days > 0:
            return f"{delta.days}天前"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}小时前"
        elif delta.seconds >= 60:
            return f"{delta.seconds // 60}分钟前"
        else:
            return f"{delta.seconds}秒前"
    except Exception:
        return "时间未知"


def filter_recent_conversations(convs, current_ts_ms: int, window_ms: int = 600_000):
    if not convs:
        return []
    try:
        current_ts = int(float(current_ts_ms))
        return [
            con for con in convs
            if isinstance(con.get("timestamp"), (int, float))
            and (current_ts - con["timestamp"]) < window_ms
        ]
    except Exception:
        return []


def format_session(session_history, srvts):
    if not session_history or not srvts:
        return "无"
    try:
        current_ts = int(float(srvts))
        recent = filter_recent_conversations(session_history, current_ts)
        if not recent:
            return "无"
        lines = []
        for idx, turn in enumerate(recent, start=1):
            user_query = turn.get("query", "")
            system_resp = turn.get("to_speak", "")
            time_ago = format_time_ago(turn["timestamp"], current_ts)
            lines.append(f"第{idx}轮({time_ago})：用户：“{user_query}”，系统：“{system_resp}”")
        return "\n".join(lines)
    except Exception:
        return "无"


def format_date(srvts):
    if not srvts or srvts in ("", "null"):
        return "无"
    try:
        ts_ms = int(float(srvts))
        dt = datetime.fromtimestamp(ts_ms // 1000, tz=timezone.utc)
        return dt.strftime("%Y%m%d")
    except Exception:
        return "无"


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
        "1. **用户意图识别**：- 结合用户历史对话信息，分析用户当前查询需求是否存在多音字、语音识别错误或用户发音不准的情况，若存在则先修复用户查询请求，并用“原始请求”和“修复请求”同等优先级分别完成后续资源筛选。\n"
        "2. **结果生成** ：- 优先返回完全匹配用户查询意图的资源SID，若无完全匹配的资源，则返回最贴近用户需求的资源SID\n"
        "3. **完全匹配判定** ：- 若候选资源列表中的条目满足“原始Query”或“修复Query”之一所包含的全部用户需求（包括资源名、别名、类别、语言、发行地区、是否付费、风格、发行日期、季数、出品方、演员、导演、角色等），则判定为完全匹配。\n\n"
        "### 输出要求\n"
        "只需要资源SID，不需要输出任何具体理由。"
    )

def build_prompt_text(query, domain, ymd, asr_gender, formatted_session):
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
        "1. **用户意图识别**：- 结合用户历史对话信息，分析用户当前查询需求是否存在多音字、语音识别错误或用户发音不准的情况，若存在则先修复用户查询请求，并用“原始请求”和“修复请求”同等优先级分别完成后续资源筛选。\n"
        "2. **结果生成** ：- 优先返回完全匹配用户查询意图的资源SID，若无完全匹配的资源，则返回最贴近用户需求的资源SID\n"
        "3. **完全匹配判定** ：- 若候选资源列表中的条目满足“原始Query”或“修复Query”之一所包含的全部用户需求（包括资源名、别名、类别、语言、发行地区、是否付费、风格、发行日期、季数、出品方、演员、导演、角色等），则判定为完全匹配。\n\n"
        "### 输出要求\n"
        "只需要输出资源的文本信息（例如：垂域：“视频”，原始标题：“潮间奇事”，主标题：“潮间奇事”，导演：“丹尼尔奈西姆、托亚弗莱瑟、卡特里奥娜麦肯齐”，演员：“马可皮科西、夏洛特贝斯特、阿隆贾库本科”，角色：“Dylan、AugieMcTeer、CoreyWelch”，标签：“综艺、剧情、奇幻”，出版时间：“2018”，付费状态：“免费”。 或者 垂域：“电台”，资源商：“蜻蜓FM”，原始标题：“滴滴故事”，主标题：“滴滴故事”，描述：“温暖 感动 正能量的故事”，主播：“佳易”，标签：“时尚生活、汽车、汽车百科”，付费状态：“免费”），不需要输出任何具体理由。"
    )


# --------------------------
# 训练数据构建
# --------------------------
def build_video_train_samples(row, id_sid_map, sid_info_map):
    query = row.get("query", "").strip()
    label_list_str = row.get("predict", "")
    if not query or not label_list_str:
        return []

    session_history = []
    if row.get("session_history"):
        try:
            session_history = json.loads(row["session_history"])
        except Exception:
            pass

    formatted_session = format_session(session_history, row.get("srvts"))
    ymd = format_date(row.get("srvts"))
    asr_gender = row.get("asr_gender", "未知")
    request_id = row.get("request_id", "")

    video_ids = [v.strip() for v in label_list_str.split(";") if v.strip()]
    sid_list = [id_sid_map.get(v) for v in video_ids if id_sid_map.get(v)]
    info_list = [sid_info_map.get(v) for v in sid_list if sid_info_map.get(v)]

    res = []
    for sid in sid_list:
        r = random.random()
        res.append({
            "instruction": build_prompt(query, "视频", ymd, asr_gender, formatted_session),
            "input": "",
            "output": sid,
            "rid": request_id
        })
    
    for info in info_list:
        r = random.random()
        if r < 0.3:
          res.append({
                "instruction": build_prompt_text(query, "视频", ymd, asr_gender, formatted_session),
                "input": "",
                "output": info,
                "rid": request_id
            }) 
         
    return res

def build_station_train_samples(row, id_sid_map, sid_info_map):
    query = row.get("query", "").strip()
    llm_label = row.get("llm_label", "")
    try:
        label_list_str = llm_label["selected_id"]
    except:
        return []
    if not query or not label_list_str:
        return []

    session_history = []
    srvts = None
    if row.get("session_history") and row.get("session_history") != "[]":
        try:
            session_history = json.loads(row["session_history"])
            srvts = int(session_history[0]["timestamp"]) + 10000
        except Exception:
            pass

    formatted_session = format_session(session_history, srvts)
    ymd = format_date(srvts)
    asr_gender = row.get("asr_gender", "未知")
    request_id = row.get("request_id", "")

    meta_ids = [v.strip() for v in label_list_str.split(";") if v.strip()]
    sid_list = [id_sid_map.get(v) for v in meta_ids if id_sid_map.get(v)]
    info_list = [sid_info_map.get(v) for v in sid_list if sid_info_map.get(v)]

    res = []
    for sid in sid_list:
        res.append({
            "instruction": build_prompt(query, "电台", ymd, asr_gender, formatted_session),
            "input": "",
            "output": sid,
            "rid": request_id
        })

    for info in info_list:
        r = random.random()
        if r < 0.3:
          res.append({
                "instruction": build_prompt_text(query, "电台", ymd, asr_gender, formatted_session),
                "input": "",
                "output": info,
                "rid": request_id
            })
    return res


# --------------------------
# 评估数据构建
# --------------------------
def build_video_eval_sample(row):
    query = row.get("query", "").strip()
    if not query:
        return None

    session_history = []
    if row.get("session_history"):
        try:
            session_history = json.loads(row["session_history"])
        except Exception:
            pass

    formatted_session = format_session(session_history, row.get("srvts"))
    ymd = format_date(row.get("srvts"))
    asr_gender = row.get("asr_gender", "未知")
    request_id = row.get("request_id", "")

    return {
        "instruction": build_prompt(query, "视频", ymd, asr_gender, formatted_session),
        "input": "",
        "output": "",
        "rid": request_id
    }

def build_station_eval_sample(row):
    query = row.get("query", "").strip()
    if not query:
        return None

    session_history = []
    if row.get("session_history"):
        try:
            session_history = json.loads(row["session_history"])
        except Exception:
            pass

    formatted_session = format_session(session_history, row.get("srvts"))
    ymd = format_date(row.get("srvts"))
    asr_gender = row.get("asr_gender", "未知")
    request_id = row.get("request_id", "")

    return {
        "instruction": build_prompt(query, "电台", ymd, asr_gender, formatted_session),
        "input": "",
        "output": "",
        "rid": request_id
    }

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

def split_station_dataset(file_path):
    station_train = []
    station_eval = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = random.random()
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if r < STATION_EVAL_RATIO:
                    station_eval.append(row)
                else:
                    station_train.append(row)
            except Exception as e:
                print(f"跳过无法解析的 JSON 行: {e} | {line[:200]}")
                continue
    return station_train, station_eval

# --------------------------
# 主流程
# --------------------------
def main():
    id_sid_map, sid_info_map = load_id_sid_map(SID_PATH)
    station_train, station_eval = split_station_dataset(STATION_INPUT_PATH)
    station_train_ext, station_eval_ext = split_station_dataset(STATION_INPUT_PATH_EXT)
    # station_train, station_eval = [], []

    train_samples = []
    with open(VIDEO_TRAIN_PATH, 'r', encoding='utf-8') as f:
        read