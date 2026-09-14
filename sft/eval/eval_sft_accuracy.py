#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import logging
import os
import csv
import datetime
from datetime import timezone
from pathlib import Path
import random
import time
import pickle
import hashlib
import base64
import struct
import urllib
import requests
import concurrent.futures
from enum import Enum
import re

# ======================================================
# 基础配置
# ======================================================
csv.field_size_limit(sys.maxsize)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dataname = "rqvae_opq_20260208_station_video_text_format_4B"

EVAL_CSV_PATH = "/lizhaoxuan1/train_data/eval_dig.csv"
SID_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"
SAMPLING_PREDICT_PATH = f"/lizhaoxuan1/llm_output/sft_single_{dataname}_infer_sampling.jsonl"

OUTPUT_CSV_PATH = f"/lizhaoxuan1/llm_output/sft_single_{dataname}_llm_judge.csv"

# ======================================================
# 大模型 API 配置
# ======================================================
api = "http://apiadmin-preview4test.ai.srv/ares/api/v1/chat/answer"
app_id = "700892433633906688"
app_key = "C34Hc5P1lkKIfF7St6+xUA=="
scope = "10013"
sid = "ai-service"
auth_url = "https://iauth.pt.xiaomi.com"
source_id = "垂域大模型标注"

headers = {"Content-Type": "application/json;charset=UTF-8"}

class Model(Enum):
    DeepSeek = "deepSeekReasoner"

# ======================================================
# 鉴权 & 请求工具
# ======================================================
cache = {}

def is_obsolete(entry, duration):
    return time.time() - entry['time'] > duration

def compute_key(func, args, kw):
    return hashlib.sha1(pickle.dumps((func.__name__, args, kw))).hexdigest()

def memorize(duration=600):
    def _wrap(func):
        def _inner(*args, **kw):
            key = compute_key(func, args, kw)
            if key in cache and not is_obsolete(cache[key], duration):
                return cache[key]["value"]
            val = func(*args, **kw)
            cache[key] = {"value": val, "time": time.time()}
            return val
        return _inner
    return _wrap

@memorize(600)
def get_token():
    t = int(time.time())
    rd = random.getrandbits(64)
    nonce = base64.b64encode(
        struct.pack(">Q", rd) + struct.pack(">L", int(t / 60))
    ).decode()

    params = f"appId={app_id}&nonce={nonce}&scope={scope}&sid={sid}"
    joind = f"GET&/token/getToken&{params}&{app_key}"
    sign = base64.b64encode(hashlib.sha1(joind.encode()).digest())

    url = f"{auth_url}/token/getToken"
    payload = {
        "appId": app_id,
        "nonce": nonce,
        "scope": scope,
        "sid": sid,
        "_sign": sign
    }
    ret = requests.get(url, params=payload)
    token = json.loads(ret.text)["data"]["token"]
    return urllib.parse.quote_plus(token)

def request_llm(prompt: str):
    token = get_token()
    url = f"{api}?appId={app_id}&token={token}&requestId=mock"

    data = {
        "model": Model.DeepSeek.value,
        "topP": 0.8,
        "temperature": 0,
        "prompt": prompt,
        "history": [],
        "sourceId": source_id
    }

    for _ in range(5):
        try:
            resp = requests.post(url, headers=headers, json=data)
            content = json.loads(resp.content.decode("utf-8"))
            res = content["data"]["content"]
            reasoning = content["data"].get("reasoning", "")
            return res.strip(), reasoning.strip()
        except Exception as e:
            time.sleep(1)

    return "Error", "Error"

def sanitize_for_csv(text: str, max_len: int = 1000) -> str:
    """
    清洗字符串
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    if max_len and len(text) > max_len:
        text = text[:max_len] + "..."

    return text.strip()

# ======================================================
# 数据加载
# ======================================================
def load_sid_map(path):
    sid_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            sid = re.sub(r"[\[\]'，, ]", "", "".join(r.get("indices", "")))
            meta = r.get("json_format") or r.get("meta_json")
            if sid and meta:
                sid_map[sid] = meta
    return sid_map

def load_predict_map(predict_file, sid_map):
    res = {}
    with open(predict_file) as f:
        for l in f:
            r = json.loads(l)
            docs = [sid_map[s] for s in r["predict"].split(";") if s in sid_map]
            res[r["rid"]] = docs
    return res

# ======================================================
# 构建 Judge Prompt
# ======================================================
def build_prompt(row, predict_docs):
    ymd = "无"
    if row.get("srvts"):
        ts = int(float(row["srvts"]))
        ymd = datetime.datetime.fromtimestamp(ts//1000, tz=timezone.utc).strftime("%Y%m%d")

    return {
        "rid": row["request_id"],
        "input": "\n".join([
            "你的任务是结合联网知识，判断下发资源是否满足用户查询。",
            "",
            f"- 请求日期: {ymd}",
            "",
            "### 用户查询",
            row["query"],
            "",
            "### 下发资源",
            "\n".join(f"{i+1}. {d}" for i, d in enumerate(predict_docs)),
            "",
            "只输出 YES 或 NO"
        ]),
        "original_match_level": row.get("matchlevel", ""),
        "predict_meta": "\n".join(
            json.dumps(d, ensure_ascii=False) for d in predict_docs
        )
    }

# ======================================================
# 主流程
# ======================================================
def main():
    sid_map = load_sid_map(SID_PATH)
    predict_map = load_predict_map(SAMPLING_PREDICT_PATH, sid_map)

    prompts = []
    with open(EVAL_CSV_PATH) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rid = row["request_id"]
            if rid in predict_map:
                prompts.append(build_prompt(row, predict_map[rid]))

    logging.info("前处理完成")

    # 并发 Judge
    results = {}
    good = bad = useless = 0

    def process(p):
        res, reason = request_llm(p["input"])
        judge = res.replace(" ", "").replace("\n", "")
        return p["rid"], judge, reason, p["original_match_level"], p["predict_meta"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as ex:
        for rid, judge, reason, gt, predict_meta in ex.map(process, prompts):
            results[rid] = (judge, reason, predict_meta)
            if gt == "ACCURATE" and judge == "YES":
                good += 1
            elif gt == "ACCURATE" and judge != "YES":
                bad += 1
            else:
                useless += 1

    logging.info(f"Same：Bad：Useless: {good}:{bad}:{useless}")

    # 写 CSV
    with open(EVAL_CSV_PATH) as fin, open(OUTPUT_CSV_PATH, "w", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(
            fout,
            fieldnames=reader.fieldnames + ["judge", "reason", "predict_meta"],
            delimiter="\t"
        )
        writer.writeheader()
        for row in reader:
            rid = row["request_id"]
            if rid in results:
                judge, reason, predict_meta = results[rid]
                row["judge"] = sanitize_for_csv(judge, max_len=20)
                row["reason"] = sanitize_for_csv(reason, max_len=1000)
                row["predict_meta"] = sanitize_for_csv(predict_meta, max_len=5000)

                writer.writerow(row)

    logging.info("评测流程完成")

if __name__ == "__main__":
    main()
