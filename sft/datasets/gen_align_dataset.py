import ast
import json
from typing import Optional, Dict, Any
import random

# 废弃 用sample

dataname = "station_video_4B_json_format.npy_rqopq_base.index"
INPUT_PATH = f"/lizhaoxuan1/sid/result/{dataname}.jsonl"
OUTPUT_JSONL_PATH = f"/lizhaoxuan1/train_data/align_{dataname}.jsonl"
OUTPUT_EVAL_JSONL_PATH = f"/lizhaoxuan1/train_data/align_{dataname}_eval.jsonl"

eval_size = 10000


def extract_field(text: str, marker: str) -> Optional[str]:
    """
    从一段中文说明里抽取以 marker 开头、以逗号/句号/分号结束的字段值。
    例如 marker="资源名称：" / "主名称：" / "标签：" / "出版时间：" 等。
    """
    if not text or marker not in text:
        return None
    start = text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = len(text)
    for sep in ["，", "。", "；"]:
        pos = text.find(sep, start)
        if pos != -1:
            end = min(end, pos)
    value = text[start:end].strip()
    value = value.rstrip("，。；,.; ")
    return value if value and value != "未知" else None


# ------------------ meta_json / json_format → meta_info ------------------
def build_meta_from_meta_json(meta_json: Dict[str, Any]) -> Dict[str, str]:
    """
    优先使用 meta_json / json_format 中的结构化字段构造 meta_info。
    兼容中英文 key。
    """
    if meta_json is None:
        meta_json = {}

    # 名称类
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

    # 人物类
    directors = (
        meta_json.get("directors")
        or meta_json.get("导演")
        or "NULL"
    )
    characters = (
        meta_json.get("characters")
        or meta_json.get("角色")
        or "NULL"
    )
    actors = (
        meta_json.get("actors")
        or meta_json.get("演员")
        or "NULL"
    )

    # 类型 / 垂域
    meta_type = (
        meta_json.get("type")
        or meta_json.get("垂域")
        or meta_json.get("资源类型")
        or "NULL"
    )

    # 付费
    pay_type = (
        meta_json.get("saletype")
        or meta_json.get("付费状态")
        or meta_json.get("付费类型")
        or "NULL"
    )

    # 标签
    tag = (
        meta_json.get("tags")
        or meta_json.get("标签")
        or "NULL"
    )

    # 语言
    languages = (
        meta_json.get("lanuages")  # 注意：原数据里拼写是 lanuages
        or meta_json.get("languages")
        or meta_json.get("语言")
        or "NULL"
    )

    # 出版时间 / 年份
    publish_time = (
        meta_json.get("publish_year")
        or meta_json.get("出版时间")
        or "NULL"
    )

    # 地区
    areas = (
        meta_json.get("areas")
        or meta_json.get("地区")
        or "NULL"
    )

    # 资源商 / IP
    cp = (
        meta_json.get("资源商")
        or "NULL"
    )

    # 主播
    station_artist = (
        meta_json.get("主播")
        or "NULL"
    )

    if station_artist not in ("NULL"):
        station_artist = f"主播:{station_artist}"
    else:
        station_artist = "NULL"

    # 视频演职人员
    video__artist = None
    has_director = directors not in (None, "", "NULL")
    has_actor = actors not in (None, "", "NULL")

    if has_director or has_actor:
        # 替换空值为未知
        director_value = directors if has_director else "未知"
        actor_value = actors if has_actor else "未知"
        video__artist = f"导演:{director_value}, 主演:{actor_value}"
    else:
        video__artist = None

    res = {
        "meta_origin_name": meta_origin_name or "NULL",
        "meta_main_name": meta_main_name or "NULL",
        "season": season or "0",
        "artists": station_artist or video__artist or "NULL",
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
    }

    return res


def build_dict_from_record(rec: Dict[str, Any]) -> Dict[str, str]:
    """
    根据 domain 分垂域读取：
      - domain == "video": 用 meta_json
      - domain == "station": 用 json_format
    再视情况用 text_format/meta 做补充，保持原来的效果。
    """
    domain = rec.get("domain", "")

    # 先根据 domain 读结构化信息
    base = {
        "meta_origin_name": "NULL",
        "meta_main_name": "NULL",
        "season": "0",
        "artists": "NULL",
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
    }

    if domain == "video":
        # 直接读 meta_json
        meta_json = rec.get("meta_json")
        if isinstance(meta_json, str):
            try:
                meta_json = json.loads(meta_json)
            except Exception:
                meta_json = None
        if isinstance(meta_json, dict):
            base = build_meta_from_meta_json(meta_json)

    elif domain == "station":
        # 读 json_format（JSON 字符串）
        json_fmt = rec.get("json_format")
        meta_dict = {}
        if isinstance(json_fmt, str):
            try:
                meta_dict = json.loads(json_fmt)
            except Exception:
                meta_dict = {}

        base = build_meta_from_meta_json(meta_dict)

    return base


# ------------------ 答案格式 ------------------
def build_item_from_text(meta_info: dict) -> str:
    """
    把 meta_info 转成： 名称:xxx,资源类型:yyy,导演:zzz,...
    """
    mapping = [
        ("meta_origin_name", "名称"),
        ("meta_type", "资源类型"),
        # ("artists", "演职人员"),
        # ("characters", "角色"),
        # ("actors", "演员"),
        # ("languages", "语言"),
        # ("areas", "地区"),
        # ("station_artist", "主播"),
        ("tags", "标签"),
        # ("cp", "资源商"),
    ]

    parts = []
    for key, label in mapping:
        value = meta_info.get(key, "NULL")
        if value not in ("NULL", "", None):
            parts.append(f"{label}:{value}")
        else:
            parts.append(f"{label}:未知")

    return ",".join(parts)



def build_sid(rec: Dict[str, Any]) -> str:
    indices = rec.get("indices")

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

def task_sid_to_item(full_sid: str, meta_info: dict) -> dict:
    """
    任务1: SID → 基础信息
    """
    user_content = (
        f"请根据资源SID:{full_sid}, "
        "返回它对应的名称、资源类型、标签信息。（格式为：名称:xxx,资源类型:yyy,标签:zzz）"
    )
    assistant_content = build_item_from_text(meta_info)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def task_item_to_sid(full_sid: str, meta_info: dict) -> dict:
    """
    任务2: 文本 → SID
    """
    item_text = build_item_from_text(meta_info)
    user_content = (
        f"请根据资源的文本信息:{item_text}, "
        "返回它对应的SID。"
    )

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": full_sid},
        ]
    }


def task_sid_to_name(full_sid: str, meta_info: dict) -> dict:
    """
    任务3: SID → 名称
    """
    user_content = (
        f"请根据资源SID:{full_sid}, "
        "返回它对应的名称。"
    )
    assistant_content = meta_info.get("meta_origin_name", "NULL")

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def task_sid_to_type(full_sid: str, meta_info: dict) -> dict:
    """
    任务4: SID → 资源类型
    """
    user_content = (
        f"请根据资源SID:{full_sid}, "
        "返回它对应的资源类型。"
    )
    assistant_content = meta_info.get("meta_type", "NULL")

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }

def task_sid_to_artist(full_sid: str, meta_info: dict) -> dict:
    """
    任务4: SID → artist
    """
    user_content = (
        f"请根据资源SID:{full_sid}, "
        "返回它对应的导演/主演/主播。（格式为：导演:xxx,主演:yyy 或 主播:zzz）"
    )
    assistant_content = meta_info.get("artists", "NULL")

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }

def task_sid_to_tag(full_sid: str, meta_info: dict) -> dict:
    """
    任务4: SID → tag
    """
    user_content = (
        f"请根据资源SID:{full_sid}, "
        "尽可能描述该资源的标签，如（儿童、教育、国产动画片等）"
    )
    assistant_content = meta_info.get("tags", "NULL")

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def gen_align_trained_jsonl():
    records = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f_in:
        line_num = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                rec_raw = json.loads(line)
            except Exception as e:
                print(f"跳过无法解析的 JSON 行 {line_num}: {e} | {line[:200]}")
                line_num += 1
                continue

            domain = rec_raw.get("domain", "")

            if domain == "video":
                res_text = rec_raw.get("meta") or rec_raw.get("text_format") or ""
            elif domain == "station":
                res_text = rec_raw.get("text_format") or rec_raw.get("meta") or ""
            else:
                res_text = rec_raw.get("text_format") or rec_raw.get("meta") or ""

            full_sid = build_sid(rec_raw)

            # 构造 meta_info（按 domain 分支）
            meta_info = build_dict_from_record(rec_raw)

            # 1) SID → ITEM
            record_sid_to_item = task_sid_to_item(full_sid, meta_info)
            records.append(record_sid_to_item)

            # 2) ITEM → SID
            clean_text = (
                res_text.replace("这是一个资源的信息，", "")
                        .replace("这是一个视频资源的信息，", "垂域：“视频”，")
            )
            record_item_to_sid = task_item_to_sid(full_sid, meta_info)
            records.append(record_item_to_sid)

            # 3) SID → 名称
            record_sid_to_name = task_sid_to_name(full_sid, meta_info)
            records.append(record_sid_to_name)

            # 4) SID → 资源类型
            record_sid_to_type = task_sid_to_type(full_sid, meta_info)
            records.append(record_sid_to_type)

            # 5) SID → 演职人员（导演主演/主播）
            if meta_info.get("artists") != "NULL":
                record_sid_to_artist = task_sid_to_artist(full_sid, meta_info)
                records.append(record_sid_to_artist)

             # 6) SID → 标签
            if meta_info.get("tags") != "NULL":
                record_sid_to_tag = task_sid_to_tag(full_sid, meta_info)
                records.append(record_sid_to_tag)
            
            line_num += 1
            if line_num % 1000 == 0:
                print(f"已处理 {line_num} 条原始样本，目前累计对话样本 {len(records)} 条")

    # 打乱 & 划分 train / eval
    random.shuffle(records)
    print(f"total size (all records): {len(records)}")

    if len(records) <= eval_size:
        eval_records = records
        train_records = []
    else:
        eval_records = records[:eval_size]
        train_records = records[eval_size:]

    print(f"train dataset: {len(train_records)}")
    print(f"eval dataset: {len(eval_records)}")

    # 写 train
    if train_records:
        with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f_out:
            for rec in train_records:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"train_data_path: {OUTPUT_JSONL_PATH}")

    # 写 eval
    with open(OUTPUT_EVAL_JSONL_PATH, "w", encoding="utf-8") as f_eval:
        for rec in eval_records:
            f_eval.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"eval_data_path: {OUTPUT_EVAL_JSONL_PATH}")


if __name__ == "__main__":
    gen_align_trained_jsonl()
