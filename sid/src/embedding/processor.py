import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "/mnt/wulinyang/models/Qwen3-Embedding-4B"
DEFAULT_OUTPUT_DIR = "/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings"

DOMAIN_CONFIG = {
    "station": {
        "embedding_fields": ["name", "subname", "tags", "text_format", "json_format"],
    },
    "video": {
        "embedding_fields": ["main_name", "text_format", "json_format"],
    },
}


CP_DICT = {}  # 可按需填充资源商中文名映射

_PAY_TYPE_MAP = {"PAY": 1, "FREE": 2}  # 0=未知, 1=付费, 2=免费
SPLIT_RE = re.compile(r"[、,;；]+")  # llm_tag 等标签字段分隔符切分（顿号/逗号/分号/中文分号）
_CURRENT_YEAR = 2026

def _norm_hardcode_field(key: str, val, obj: dict = None):
    """Normalize hardcode field values to their canonical integer form."""
    if key in ("saletype", "pay_type"):
        if val is None:
            return 0  # 未知
        if isinstance(val, str):
            mapped = _PAY_TYPE_MAP.get(val.strip().upper())
            if mapped is not None:
                return mapped
            try:
                return int(val)
            except Exception:
                return 0
        try:
            return int(val)
        except Exception:
            return 0
    if key == "publish_year":
        if val is None:
            return 0
        try:
            return int(val)
        except Exception:
            return 0
    if key == "years_ago":
        # 从 obj 里取 publish_year 计算距今年份差
        py = obj.get("publish_year") if obj else None
        if py is None:
            return 0
        try:
            return max(0, _CURRENT_YEAR - int(py))
        except Exception:
            return 0
    return val


# ── text_format 重新拼接工具函数 ──────────────────────────────────────────────

def _to_text(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, (list, tuple, set)):
        items = [x.strip() if isinstance(x, str) else str(x).strip() for x in v if x is not None]
        items = [i for i in items if i]
        return "、".join(items) if items else None
    s = str(v).strip()
    return s if s else None


def _parse_saletype_int(v):
    mapping = {0: "免费", 1: "付费", 2: "付费"}
    if v is None:
        return None
    if isinstance(v, int):
        return mapping.get(v)
    try:
        return mapping.get(int(str(v).strip()))
    except Exception:
        return None


def _parse_saletype_str(v):
    mapping = {"PAY": "付费", "FREE": "免费"}
    if v is None:
        return None
    return mapping.get(str(v).strip().upper())


def _parse_artist(v):
    if v is None:
        return None
    s = _to_text(v)
    if not s:
        return None
    parts = s.split("、")
    return "、".join(parts[:3])


def _merge_tags_video(doc):
    val = doc.get("llm_tag")
    if not val:
        return None
    seen, out = set(), []
    for tok in SPLIT_RE.split(str(val)):
        t = tok.strip()
        if t and t.casefold() not in seen:
            seen.add(t.casefold())
            out.append(t)
    return "、".join(out) if out else None


def _merge_tags_station(doc, keys=("f_tags", "s_tags", "k_tags", "c_tags")):
    seen, out = set(), []
    for key in keys:
        val = doc.get(key)
        if not val:
            continue
        tokens = val.split(",") if isinstance(val, str) else list(val)
        for tok in tokens:
            t = (tok.strip() if isinstance(tok, str) else str(tok).strip())
            if t and t.casefold() not in seen:
                seen.add(t.casefold())
                out.append(t)
    return "、".join(out) if out else None


def _resolve_title(*titles: Optional[str]) -> Optional[str]:
    """从多个标题里挑出最终标题。

    若标题间存在包含关系（去掉非字母数字后比较），只保留最长的一个；
    否则用「、」拼接去重。这样上游把 meta_main_name 改残的情况（如
    '琬廷语堂612岁...Lv' vs 完整 'meta_origin_name'）不会把残缺串拼进去。
    """
    cleaned = []
    for t in titles:
        s = _to_text(t)
        if s:
            cleaned.append(s)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]

    def alnum(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z一-鿿]", "", s).casefold()

    normed = [alnum(s) for s in cleaned]
    # 若任意一个标题是另一个的子串，则只保留最长的那个
    has_contain = any(
        i != j and ni and ni in nj
        for i, ni in enumerate(normed)
        for j, nj in enumerate(normed)
    )
    if has_contain:
        # 按 alnum 长度降序取最长，原串保留
        order = sorted(range(len(cleaned)), key=lambda k: len(normed[k]), reverse=True)
        return cleaned[order[0]]
    # 无包含关系，去重拼接
    seen, parts = set(), []
    for s in cleaned:
        key = s.casefold()
        if key not in seen:
            seen.add(key)
            parts.append(s)
    return "、".join(parts) if parts else None


def build_video_text_format(doc: dict) -> str:
    """重新拼接视频 text_format，去掉垂域字段。"""
    main_actors = doc.get("main_actors") or doc.get("actors")
    title = _resolve_title(doc.get("meta_main_name"), doc.get("meta_origin_name"))
    data_map = {
        "标题":     title,
        "导演":     _parse_artist(_to_text(doc.get("directors"))),
        "演员":     _parse_artist(_to_text(main_actors)),
        "角色":     _parse_artist(_to_text(doc.get("characters"))),
        "标签":     _merge_tags_video(doc),
        "描述":     _to_text(doc.get("rec_tag_desc")),
        "IP系列名": _to_text(doc.get("patchwall_ip_name")),
        "出版时间": _to_text(doc.get("publish_year")),
        "付费状态": _parse_saletype_str(doc.get("pay_type")),
    }
    ordered_keys = ["标题", "IP系列名", "标签", "付费状态", "出版时间", "导演", "演员", "角色", "描述"]
    parts = []
    for k in ordered_keys:
        v = data_map.get(k)
        if v:
            parts.append(k + "：“" + v + "”")
    return "，".join(parts) + "。" if parts else ""


def build_station_text_format(doc: dict) -> str:
    """重新拼接电台 text_format，去掉垂域字段。"""
    cp_raw = doc.get("cp")
    cp_val = CP_DICT.get(str(cp_raw).strip(), _to_text(cp_raw)) if cp_raw else None
    data_map = {
        "资源商":   cp_val,
        "原始标题": _to_text(doc.get("name")),
        "主标题":   _to_text(doc.get("mainname")),
        "副标题":   _to_text(doc.get("subname")),
        "描述":     _to_text(doc.get("desc")),
        "主播":     _to_text(doc.get("br_name")),
        "标签":     _merge_tags_station(doc),
        "付费状态": _parse_saletype_int(doc.get("saletype")),
    }
    ordered_keys = ["资源商", "原始标题", "主标题", "副标题", "描述", "主播", "标签", "付费状态"]
    parts = []
    for k in ordered_keys:
        v = data_map.get(k)
        if v:
            parts.append(k + "：“" + v + "”")
    return "，".join(parts) + "。" if parts else ""


def _resolve_domain(obj: dict) -> str:
    """从顶层 domain 或 json_format「垂域」推断域，返回 video/station。"""
    domain = obj.get("domain")
    if not domain:
        jf = obj.get("json_format")
        if jf:
            try:
                jf_dict = json.loads(jf) if isinstance(jf, str) else jf
                domain = jf_dict.get("垂域") or jf_dict.get("domain")
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    if domain in ("电台", "station"):
        return "station"
    return "video"


def rebuild_text_format(obj: dict) -> Optional[str]:
    """根据 domain 或 json_format 的「垂域」字段重新拼接 text_format。

    优先读顶层 domain；无则解析 json_format 里的「垂域」字段
    （取值「视频」/「电台」）分流。默认 video。
    """
    if _resolve_domain(obj) == "station":
        return build_station_text_format(obj)
    return build_video_text_format(obj)



@dataclass
class ShardInfo:
    start: int
    end: int


def count_lines(file_path: Path) -> int:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"Failed to count lines for {file_path}: {e}")
        return 0


def compute_shard(total: int, num_gpus: int, gpu_id: int) -> ShardInfo:
    lines_per_gpu = total // num_gpus
    start_line = gpu_id * lines_per_gpu
    end_line = start_line + lines_per_gpu if gpu_id < num_gpus - 1 else total
    return ShardInfo(start=start_line, end=end_line)


def load_model(model_name_or_path: str, device: Optional[str] = None):
    if device:
        device_obj = torch.device(device)
    else:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("sentence-transformers not installed. pip install sentence-transformers") from e

    model = SentenceTransformer(model_name_or_path, device=device_obj)
    embed_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Loaded model {model_name_or_path} on {device_obj}, dim={embed_dim}")
    return model, embed_dim, device_obj


def default_patterns():
    # Patterns expect Chinese punctuation and optional Chinese quotes around content
    return {
        "main_title": re.compile(r"(主标题：\“?.*?\”?)(?=[，。,]|$)", flags=re.UNICODE),
        "tags": re.compile(r"标签：\“?(.*?)\”?[,，]?", flags=re.UNICODE),
        "subname": re.compile(r"副标题：\“?(.*?)\”?[,，]?", flags=re.UNICODE),
    }


def extract_fields(obj: dict, patterns: Optional[Dict[str, re.Pattern]] = None) -> Dict[str, Optional[str]]:
    pats = patterns or default_patterns()
    text_fmt = obj.get("text_format")
    extracted = {
        "tags": None,
        "subname": None,
        "main_name_extracted": None,
    }
    if isinstance(text_fmt, str):
        m_tags = pats["tags"].search(text_fmt)
        if m_tags:
            extracted["tags"] = m_tags.group(1)
        m_sub = pats["subname"].search(text_fmt)
        if m_sub:
            extracted["subname"] = m_sub.group(1)
        m_main = pats["main_title"].search(text_fmt)
        if m_main:
            extracted["main_name_extracted"] = m_main.group(1)
    return extracted


def resolve_main_name(obj: dict, extracted: Dict[str, Optional[str]]) -> Optional[str]:
    return extracted.get("main_name_extracted") or obj.get("meta_main_name") or obj.get("meta_origin_name") or obj.get("name")


def prepare_texts(obj: dict, fields: List[str], domain: str, patterns=None, rebuild: bool = False) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    extracted = extract_fields(obj, patterns)
    main_name = resolve_main_name(obj, extracted)
    texts: Dict[str, Optional[str]] = {}
    for field in fields:
        if rebuild and field == "text_format":
            texts[field] = rebuild_text_format(obj)
            continue
        if field == "main_name":
            texts[field] = main_name
        elif field == "name" and domain == "station":
            # station name prefer explicit name/rawname/mainname order
            texts[field] = obj.get("name") or obj.get("mainname") or obj.get("rawname")
        elif field == "tags":
            texts[field] = extracted.get("tags")
        elif field == "subname":
            texts[field] = extracted.get("subname") or obj.get("subname")
        else:
            val = obj.get(field)
            texts[field] = val if isinstance(val, str) else None
    return texts, extracted


def embed_texts(model, texts: List[str], batch_size: int, max_seq_length: int, progress_desc: str = "") -> np.ndarray:
    # sentence_transformers encode handles batching internally
    if progress_desc:
        logger.info(f"{progress_desc}: encoding {len(texts)} texts, batch_size={batch_size}")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    # model may already handle truncation; ensure max_seq_length passed via model config if needed
    return emb.astype(np.float32, copy=False)


def process_shard(
    input_path: str,
    output_dir: str,
    model_name_or_path: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    embedding_fields: Optional[List[str]] = None,
    hardcode_fields: Optional[List[str]] = None,
    batch_size: int = 32,
    max_seq_length: int = 512,
    max_rows: Optional[int] = None,
    num_gpus: int = 1,
    gpu_id: int = 0,
    domain: str = "video",
    rebuild: bool = False,
) -> Tuple[Path, Dict[str, Path]]:
    if embedding_fields is None:
        embedding_fields = DOMAIN_CONFIG.get(domain, {}).get("embedding_fields", ["main_name", "text_format", "json_format"])
    hardcode_fields = hardcode_fields or []

    input_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_lines = count_lines(input_file)
    shard = compute_shard(total_lines, num_gpus, gpu_id)
    logger.info(f"GPU {gpu_id}/{num_gpus}: processing lines [{shard.start}, {shard.end}) of {total_lines}")

    model, embed_dim, device_obj = load_model(model_name_or_path, device)

    # collect texts per field
    field_texts: Dict[str, List[Optional[str]]] = {f: [] for f in embedding_fields}
    metas = []

    with input_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < shard.start:
                continue
            if idx >= shard.end:
                break
            if max_rows is not None and len(metas) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            texts, extracted = prepare_texts(obj, embedding_fields, domain, rebuild=rebuild)
            for k, v in texts.items():
                field_texts[k].append(v)

            # 原始数据原样保留，仅 text_format 用 rebuild 函数的新值覆盖
            meta_obj = dict(obj)
            if rebuild:
                new_tf = rebuild_text_format(obj)
                if new_tf is not None:
                    meta_obj["text_format"] = new_tf
            metas.append(meta_obj)

    logger.info(f"GPU {gpu_id}: loaded {len(metas)} items to embed")

    # embed per field with missing -> zeros
    field_arrays: Dict[str, np.ndarray] = {}
    for field, vals in field_texts.items():
        if not vals:
            continue
        texts = [v if (isinstance(v, str) and v.strip()) else None for v in vals]
        to_encode_idx = [i for i, v in enumerate(texts) if v is not None]
        if not to_encode_idx:
            emb_matrix = np.zeros((len(vals), embed_dim), dtype=np.float32)
        else:
            to_encode = [texts[i] for i in to_encode_idx]
            embeds = embed_texts(
                model,
                to_encode,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                progress_desc=f"gpu{gpu_id}:{field}",
            )
            emb_matrix = np.zeros((len(vals), embeds.shape[1]), dtype=np.float32)
            for pos, enc_idx in enumerate(to_encode_idx):
                emb_matrix[enc_idx] = embeds[pos]
        field_arrays[field] = emb_matrix
        logger.info(f"GPU {gpu_id}: field {field} -> shape {emb_matrix.shape}")

    # save shard outputs
    shard_paths: Dict[str, Path] = {}
    for field, arr in field_arrays.items():
        out_file = out_dir / f"{field}_embedding_gpu{gpu_id}.npy"
        np.save(out_file, arr)
        shard_paths[field] = out_file

    meta_file = out_dir / f"meta_gpu{gpu_id}.jsonl"
    with meta_file.open("w", encoding="utf-8") as mf:
        for m in metas:
            mf.write(json.dumps(m, ensure_ascii=False) + "\n")

    return meta_file, shard_paths


def merge_shards(output_dir: str, fields: List[str], num_gpus: int, delete_partials: bool = True):
    out_dir = Path(output_dir)
    merged_meta = out_dir / "meta.jsonl"
    # merge meta
    with merged_meta.open("w", encoding="utf-8") as fout:
        for gid in range(num_gpus):
            shard = out_dir / f"meta_gpu{gid}.jsonl"
            if not shard.exists():
                raise FileNotFoundError(f"Missing meta shard: {shard}")
            with shard.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    # merge embeddings
    for field in fields:
        shards = []
        for gid in range(num_gpus):
            path = out_dir / f"{field}_embedding_gpu{gid}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing embedding shard: {path}")
            shards.append(np.load(path))
        merged = np.concatenate(shards, axis=0)
        out_path = out_dir / f"{field}_embedding.npy"
        np.save(out_path, merged)
    if delete_partials:
        for gid in range(num_gpus):
            meta_shard = out_dir / f"meta_gpu{gid}.jsonl"
            if meta_shard.exists():
                meta_shard.unlink()
            for field in fields:
                emb_shard = out_dir / f"{field}_embedding_gpu{gid}.npy"
                if emb_shard.exists():
                    emb_shard.unlink()
    logger.info(f"Merged shards into {output_dir}, partials deleted={delete_partials}")


# CLI helper for standalone use

def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate embeddings with sharded multi-GPU support")
    p.add_argument("--input", required=True, help="Input JSONL file")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory (default: %(default)s)")
    p.add_argument("--domain", choices=list(DOMAIN_CONFIG.keys()), default="video", help="Domain of input: station or video")
    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL, help="Embedding model path/name")
    p.add_argument("--device", default=None, help="Device, e.g., cuda:0; default auto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--embedding_fields", type=str, default=None, help="Comma-separated embedding fields; default per domain")
    p.add_argument("--hardcode_fields", type=str, default=None, help="Comma-separated extra meta fields to keep (no default)")
    p.add_argument("--merge", action="store_true", help="Only merge shards and exit")
    p.add_argument("--delete_partials", action="store_true", help="Delete shard files after merge")
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--rebuild_text_format", action="store_true", help="Rebuild text_format from meta (去垂域前缀+标题去重) and normalize hardcode fields, for video_data_v2 pipeline")
    return p


def main_cli():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    fields = None
    if args.embedding_fields:
        fields = [f.strip() for f in args.embedding_fields.split(',') if f.strip()]
    hard_fields = [] if args.hardcode_fields is None else [f.strip() for f in args.hardcode_fields.split(',') if f.strip()]

    if args.merge:
        if fields is None:
            fields = DOMAIN_CONFIG.get(args.domain, {}).get("embedding_fields", [])
        merge_shards(args.output_dir, fields=fields, num_gpus=args.num_gpus, delete_partials=args.delete_partials)
        return

    process_shard(
        input_path=args.input,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        embedding_fields=fields,
        hardcode_fields=hard_fields,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_rows=args.max_rows,
        num_gpus=args.num_gpus,
        gpu_id=args.gpu_id,
        domain=args.domain,
        rebuild=args.rebuild_text_format,
    )


if __name__ == "__main__":
    main_cli()
