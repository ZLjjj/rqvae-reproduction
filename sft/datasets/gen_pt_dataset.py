#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import ast

# dataname = "merged_emb.npy_rqopq_base_3rq_1opq2_bigemb"
# dataname = "station_video_4B_json_format.npy_rqopq_base"
# dataname = "rqvae_opq_20260208_station_video_text_format_4B"
dataname = "20260204_kmeans_8_2enhanced.5layers"

INPUT_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"
OUTPUT_FILE_NAME = f"pt_{dataname}"
OUTPUT_JSONL_PATH = f"/lizhaoxuan1/train_data/{OUTPUT_FILE_NAME}.jsonl"
DATASET_INFO_PATH = f"/lizhaoxuan1/gensearchrec/llama-factory/data/dataset_info.json"
# ======================================================================

def build_sid_from_indices(rec):
    indices = rec.get("indices") or rec.get("sid")

    if isinstance(indices, list):
        cleaned = [str(x).strip() for x in indices if x not in (None, "", " ")]
        if cleaned:
            return "".join(cleaned)

    if isinstance(indices, str):
        # 尝试把字符串解析成 list
        try:
            arr = ast.literal_eval(indices)
            if isinstance(arr, list):
                cleaned = [str(x).strip() for x in arr if x not in (None, "", " ")]
                return "".join(cleaned)
        except Exception:
            # 解析失败就用原始字符串
            return indices.strip()

    return "UNKNOWN_SID"


def pick_res_text(rec):
    domain = rec.get("domain", "")

    # if domain == "video":
    #     info = rec.get("meta") or ""
    #     if info != "":
    #         info = info.replace("这是一个视频资源的信息，", "垂域：视频")
    #     return info

    # if domain == "station":
    #     info = rec.get("text_format") or ""
    #     if info != "":
    #         info = info.replace("“", "").replace("”", "")
    #     return info
    if domain == "station" or domain == "video":
        info = rec.get("text_format") or ""
        if info != "":
            info = info.replace("“", "").replace("”", "")
        return info

    # 其他兜底
    return rec.get("text_format") or rec.get("json_format") or ""


def gen_pre_trained_jsonl():
    line_num = 0
    valid_num = 0

    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line_num += 1
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"跳过无法解析的 JSON 行 {line_num}: {e} | {line[:200]}")
                continue

            res_text = pick_res_text(rec).strip()
            if not res_text:
                print(f"跳过无资源文本的行 {line_num}")
                continue

            full_sid = build_sid_from_indices(rec)

            json_data = {
                "text": f"这是一条资源的信息：{res_text} 该资源的SID为：{full_sid}"
                # "text": f"SID为：{full_sid}，该SID表示的资源信息为：{res_text}"
            }

            f_out.write(json.dumps(json_data, ensure_ascii=False) + '\n')
            valid_num += 1

            if valid_num % 1000 == 0:
                print(f"已处理有效数据 {valid_num} 条（读取行数 {line_num}）")

    print(f"输出文件: {OUTPUT_JSONL_PATH}")
    print(f"总读取行数: {line_num}，有效数据 {valid_num} 条")


def update_dataset_info(key, file_path, config_path):
    item = {
        "file_name": file_path,
        "columns": {
            "prompt": "text"
        }
    }

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # update
    if key in data:
        print(f"[INFO] Key '{key}' already exists, will be overwritten.")
    else:
        print(f"[INFO] Adding new key '{key}' to dataset_info.json")

    data[key] = item

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] dataset_info.json updated.")


if __name__ == "__main__":
    gen_pre_trained_jsonl()
    update_dataset_info(OUTPUT_FILE_NAME, OUTPUT_JSONL_PATH, DATASET_INFO_PATH)
