import ast
import json
from typing import Optional, Dict, Any
import random
from collections import defaultdict

# ========= split config =========
VAL_SIZE = 2000         
TEST_SIZE = 2000 
TEST_EACH_DOMAIN_BY_LEFTOVER = True

VIDEO_SAMPLE_RATIO = 0.99      # video 进 train/val 的比例
STATION_SAMPLE_RATIO = 0.95    # station 进 train/val 的比例
OTHER_SAMPLE_RATIO = 0.0

# dataname = "station_video_4B_json_format.npy_rqopq_base"
# dataname = "rqvae_opq_20260208_station_video_text_format_4B"
dataname = "vs_Qwen4B_rqvae_hc_ema"

SID_PATH = f"/lizhaoxuan1/sid/result/{dataname}.index.meta.jsonl"

TRAIN_FILE_NAME = f"align_train_station_{STATION_SAMPLE_RATIO}_video_{VIDEO_SAMPLE_RATIO}_{dataname}_v2"
VAL_FILE_NAME = f"align_val_station_{STATION_SAMPLE_RATIO}_video_{VIDEO_SAMPLE_RATIO}_{dataname}_v2"
VAL_EXT_FILE_NAME = f"align_val_ext_station_{STATION_SAMPLE_RATIO}_video_{VIDEO_SAMPLE_RATIO}_{dataname}_v2"
TEST_FILE_NAME = f"align_test_station_{STATION_SAMPLE_RATIO}_video_{VIDEO_SAMPLE_RATIO}_{dataname}_v2"

OUTPUT_TRAIN_JSONL_PATH = f"/lizhaoxuan1/train_data/{TRAIN_FILE_NAME}.jsonl"
OUTPUT_VAL_JSONL_PATH   = f"/lizhaoxuan1/train_data/{VAL_FILE_NAME}.jsonl"
OUTPUT_VAL_EXT_JSONL_PATH   = f"/lizhaoxuan1/train_data/{VAL_EXT_FILE_NAME}.jsonl"
OUTPUT_TEST_JSONL_PATH  = f"/lizhaoxuan1/train_data/{TEST_FILE_NAME}.jsonl"

DATASET_INFO_PATH = f"/lizhaoxuan1/gensearchrec/llama-factory/data/dataset_info.json"

random.seed(2026)
# parse general dataset
def convert_belle_to_align_format(input_path, output_path):
    import json

    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            instruction = data.get("instruction", "")
            input_text = data.get("input", "")
            output = data.get("output", "")

            user_text = instruction
            if input_text:
                user_text += "\n" + input_text

            item = {
                "task_type": "general_instruction",
                "domain": "general",
                "messages": [
                    {"role": "user", "content": user_text.strip()},
                    {"role": "assistant", "content": output.strip()}
                ]
            }

            results.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Belle converted: {len(results)}")
    return results


def convert_sharegpt_to_align_format(input_path, output_path):
    import json

    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        conversations = sample.get("conversations", [])

        messages = []
        for conv in conversations:
            role = conv.get("from")
            content = conv.get("value", "")

            if role == "human":
                messages.append({"role": "user", "content": content})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": content})

        if len(messages) >= 2:
            results.append({
                "task_type": "multi_turn_chat",
                "domain": "general",
                "messages": messages
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"ShareGPT converted: {len(results)}")
    return results


def merge_datasets(*input_records):
    import json
    import random

    all_data = []

    for records in input_records:
        all_data.extend(records)

    random.shuffle(all_data)

    print(f"Final dataset size: {len(all_data)}")
    return all_data

# ------------------ meta_json / json_format → meta_info ------------------
def build_meta_from_meta_json(meta_json: Dict[str, Any]) -> Dict[str, str]:
    if meta_json is None:
        meta_json = {}

    meta_origin_name = (
        meta_json.get("name")
        or meta_json.get("原始标题")
        or meta_json.get("原始名称")
        or "NULL"
    )
    meta_main_name = (
        meta_json.get("mainname")
        or meta_json.get("主标题")
        or meta_json.get("主名称")
        or meta_origin_name
        or "NULL"
    )

    season = meta_json.get("season", "0") or "0"

    directors = meta_json.get("directors") or meta_json.get("导演") or "NULL"
    characters = meta_json.get("characters") or meta_json.get("角色") or "NULL"
    actors = meta_json.get("actors") or meta_json.get("演员") or "NULL"

    meta_type = meta_json.get("type") or meta_json.get("垂域") or meta_json.get("资源类型") or "NULL"
    pay_type = meta_json.get("saletype") or meta_json.get("付费状态") or meta_json.get("付费类型") or "NULL"
    tag = meta_json.get("tags") or meta_json.get("标签") or "NULL"

    languages = meta_json.get("lanuages") or meta_json.get("languages") or meta_json.get("语言") or "NULL"
    publish_time = meta_json.get("publish_year") or meta_json.get("出版时间") or "NULL"
    areas = meta_json.get("areas") or meta_json.get("地区") or "NULL"
    cp = meta_json.get("资源商") or meta_json.get("出版商") or "NULL"
    station_artist = meta_json.get("主播") or "NULL"
    desc = meta_json.get("描述") or "NULL"

    ip = meta_json.get("IP系列名") or "NULL"

    return {
        "meta_origin_name": meta_origin_name or "NULL",
        "meta_main_name": meta_main_name or "NULL",
        "season": season or "0",
        "directors": directors or "NULL",
        "characters": characters or "NULL",
        "actors": actors or "NULL",
        "meta_type": meta_type or "NULL",
        "pay_type": pay_type or "NULL",
        "tags": tag or "NULL",
        "languages": languages or "NULL",
        "publish_time": publish_time or "NULL",
        "areas": areas or "NULL",
        "station_artist": station_artist or "NULL",
        "cp": cp or "NULL",
        "desc": desc or "NULL",
        "ip" : ip or "NULL",
    }

def build_dict_from_record(rec: Dict[str, Any]) -> Dict[str, str]:
    domain = rec.get("domain", "")
    base = {
        "meta_origin_name": "NULL",
        "meta_main_name": "NULL",
        "season": "0",
        "directors": "NULL",
        "characters": "NULL",
        "actors": "NULL",
        "meta_type": "NULL",
        "pay_type": "NULL",
        "tags": "NULL",
        "languages": "NULL",
        "publish_time": "NULL",
        "areas": "NULL",
        "station_artist": "NULL",
        "cp": "NULL",
        "desc": "NULL",
        "domain": "NULL",
        "ip": "NULL"
    }

    # if domain == "video":
    #     # meta_json = rec.get("meta_json")
    #     meta_json = rec.get("json_format")
    #     if isinstance(meta_json, str):
    #         try:
    #             meta_json = json.loads(meta_json)
    #         except Exception:
    #             meta_json = None
    #     if isinstance(meta_json, dict):
    #         base = build_meta_from_meta_json(meta_json)
    #         base["domain"] = "video"

    # elif domain == "station":
    #     json_fmt = rec.get("json_format")
    #     meta_dict = {}
    #     if isinstance(json_fmt, str):
    #         try:
    #             meta_dict = json.loads(json_fmt)
    #         except Exception:
    #             meta_dict = {}
    #     base = build_meta_from_meta_json(meta_dict)
    #     base["domain"] = "station"

    if domain == "station" or domain == "video":
        json_fmt = rec.get("json_format") or rec.get("meta_json")
        meta_dict = {}
        if isinstance(json_fmt, str):
            try:
                meta_dict = json.loads(json_fmt)
            except Exception:
                meta_dict = {}
        else:
            meta_dict = json_fmt
        base = build_meta_from_meta_json(meta_dict)
        base["domain"] = domain

    return base

def build_item_from_text(meta_info: dict) -> str:
    mapping = [
        ("meta_origin_name", "资源名"),
        ("meta_type", "资源类型"),
        ("tags", "标签"),
    ]
    parts = []
    for key, label in mapping:
        value = meta_info.get(key, "NULL")
        if value in ("NULL", "", None):
            continue
        parts.append(f"{label}:{value}")
    return ",".join(parts)

def build_attr_from_text(meta_info: dict) -> str:
    mapping = [
        ("meta_type", "资源类型"),
        ("directors", "导演"),
        ("actors", "主演"),
        ("station_artist", "主播"),
        ("tags", "标签"),
        ("desc", "描述"),
        ("pay_type", "付费类型"),
        ("cp", "出版商"),
        ("publish_time", "出版年份"),
    ]
    parts = []
    for key, label in mapping:
        value = meta_info.get(key, "NULL")
        if value in ("NULL", "", None):
            continue
        parts.append(f"{label}:{value}")
    random.shuffle(parts)
    return ",".join(parts)

def build_sid(rec: Dict[str, Any]) -> str:
    indices = rec.get("sid") or rec.get("indices")
    if isinstance(indices, list):
        return "".join(str(x) for x in indices)
    if isinstance(indices, str):
        try:
            arr = ast.literal_eval(indices)
            if isinstance(arr, list):
                return "".join(str(x) for x in arr)
        except Exception:
            return indices
    return "UNKNOWN_SID"

def shuffle_tags(text: str):
    if not text:
        return ""
    tags = {tag.strip() for tag in text.split("、") if tag.strip()}
    tags = list(tags)
    if not tags:
        return ""  
    random.shuffle(tags)
    return "、".join(tags)

def shuffle_artists(text: str):
    if not text:
        return ""
    artists = {artist.strip() for artist in text.split("、") if artist.strip()}
    artists = list(artists)
    if not artists:
        return ""  
    random.shuffle(artists)
    return "、".join(artists)

def choose_template(templates, **kwargs):
    tpl = random.choice(templates)
    return tpl.format(**kwargs)

# ------------------ tasks sid to xx ------------------
def task_sid_to_item(full_sid: str, meta_info: dict) -> dict:

    templates = [
        "请根据资源SID:{sid} 返回名称、资源类型、标签。（格式：资源名:xxx,资源类型:yyy,标签:zzz）",
        "{sid} 的名称、资源类型和标签是什么？",
        "{sid}，输出资源名、资源类型和标签信息。",
        "SID:{sid} 对应的资源信息（名称、类型、标签）是什么？",
        "根据SID:{sid} 返回资源基本信息（名称、资源类型、标签）。"
    ]

    user_content = choose_template(templates, sid=full_sid)

    return {
        "task_type": "sid_to_item",
        "domain": meta_info.get("domain", "NULL"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": build_item_from_text(meta_info)}
        ]
    }

def task_sid_to_name(full_sid: str, meta_info: dict) -> dict:

    templates = [
        "请根据资源SID:{sid} 返回资源名。",
        "{sid}的名称",
        "{sid}输出资源名称。",
        "资源SID:{sid} 的名字是？",
        "SID:{sid} 的资源名是什么？"
        "{sid}的标题"
    ]

    user_content = choose_template(templates, sid=full_sid)

    return {
        "task_type": "sid_to_name",
        "domain": meta_info.get("domain", "NULL"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": meta_info.get("meta_origin_name", "NULL")}
        ]
    }

def task_sid_to_artist(full_sid: str, artists: str, title: str, domain: str) -> dict:

    templates = [
        "请根据资源SID:{sid} 返回{title}。（格式：{title}:xxx）",
        "{sid} 的{title}是谁？",
        "{sid}，输出{title}信息。",
        "资源{sid} 对应的{title}是？",
        "{sid} 的{title}是谁？（格式：{title}:xxx）"
    ]

    user_content = choose_template(
        templates,
        sid=full_sid,
        title=title
    )

    return {
        "task_type": "sid_to_artist",
        "domain": domain,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"{title}:{artists}"}
        ]
    }

def task_sid_to_tag(full_sid: str, meta_info: dict) -> dict:

    templates = [
        "请根据资源SID:{sid} 返回标签。",
        "{sid} 的标签有哪些",
        "给定SID:{sid}，输出资源标签。",
        "{sid} 的标签",
        "SID:{sid} 的资源标签内容是？"
    ]

    user_content = choose_template(templates, sid=full_sid)

    tags = meta_info.get("tags", "NULL")
    tag = shuffle_tags(tags)

    return {
        "task_type": "sid_to_tag",
        "domain": meta_info.get("domain", "NULL"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": tag}
        ]
    }

def task_sid_to_domain(full_sid: str, meta_info: dict) -> dict:

    templates = [
        "请根据资源SID:{sid} 判断资源类型（电台/视频）。",
        "{sid} 是视频还是电台？",
        "{sid}的资源类型",
        "SID:{sid} 的类型是视频还是电台？",
    ]

    user_content = choose_template(templates, sid=full_sid)

    domain = "视频" if meta_info.get("domain") == "video" else "电台"

    return {
        "task_type": "sid_to_domain",
        "domain": meta_info.get("domain", "NULL"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": domain}
        ]
    }

def task_sid_to_cp(full_sid: str, meta_info: dict) -> dict:

    templates = [
        "请根据资源SID:{sid} 返回出版商。",
        "{sid} 的出版商",
        "给定SID:{sid}，输出出版商。",
        "资源{sid} 对应的出版商是？",
        "{sid}的资源出版商是什么？"
    ]

    user_content = choose_template(templates, sid=full_sid)

    return {
        "task_type": "sid_to_cp",
        "domain": meta_info.get("domain", "NULL"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": meta_info.get("cp", "NULL")}
        ]
    }

def task_sid_to_year(full_sid: str, meta_info: dict) -> dict:

    templates = [
        "请根据资源SID:{sid} 返回年份。",
        "{sid}是哪一年发布的？",
        "{sid}是几几年的",
        "资源{sid} 的年份是？",
        "SID:{sid} 对应的发布年份是什么？"
    ]

    user_content = choose_template(templates, sid=full_sid)

    return {
        "task_type": "sid_to_year",
        "domain": meta_info.get("domain", "NULL"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": meta_info.get("publish_time", "NULL")}
        ]
    }

def task_sid_to_paytype(full_sid: str, meta_info: dict) -> dict:

    templates = [
        "请根据资源SID:{sid} 判断付费类型（付费/免费）。",
        "{sid}是付费还是免费资源？",
        "给定SID:{sid}，判断付费类型。",
        "资源{sid}的付费类型",
        "SID:{sid}是付费还是免费？"
    ]

    user_content = choose_template(templates, sid=full_sid)

    return {
        "task_type": "sid_to_paytype",
        "domain": meta_info.get("domain", "NULL"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": meta_info.get("pay_type", "NULL")}
        ]
    }

def task_sid_to_ip(full_sid: str, meta_info: dict) -> dict:

    templates = [
        "请根据资源SID:{sid} 返回IP系列名。",
        "{sid}属于哪个IP？",
        "给定SID:{sid}，输出IP系列",
        "资源{sid}的IP名",
        "SID:{sid}对应的IP系列是什么？"
    ]

    user_content = choose_template(templates, sid=full_sid)

    return {
        "task_type": "sid_to_ip",
        "domain": meta_info.get("domain", "NULL"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": meta_info.get("ip", "NULL")}
        ]
    }

# ------------------ tasks xx to sid ------------------
def task_attr_to_sid(full_sid: str, meta_info: dict) -> dict:
    