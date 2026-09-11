# Embedding Format

## Input (JSONL)
- One object per line.
- Use `--domain station` or `--domain video` to choose default field handling. You can override embedding fields via `--embedding_fields`.

### station fields (expected)
- id
- name/rawname/mainname (name prefer order: name > mainname > rawname)
- subname
- tags (extracted from `text_format` by regex: `标签：`…)
- text_format (string)
- json_format (string)
- other meta fields (cpId/type/cp/desc/f_tags/s_tags/ctags/br_name/saletype/ison/cover/… will be kept in meta, not embedded by default)

### video fields (expected)
- id
- meta_main_name / meta_origin_name / name (used to resolve main_name)
- text_format (string)
- json_format (string)
- other meta fields (directors/staff/actors/languages/season/publish_year/… kept in meta, not embedded by default)

## Embeddings
- Per-field `.npy` files (float32), one matrix per embedding field.
- Sharded outputs (per GPU): `{field}_embedding_gpu{gid}.npy`
- Merged outputs: `{field}_embedding.npy` after `--merge` (concatenate shards by row order).
- Default fields per domain:
  - station: `name,subname,tags,text_format,json_format`
  - video: `main_name,text_format,json_format`

## Meta (jsonl)
- Sharded: `meta_gpu{gid}.jsonl`; merged: `meta.jsonl`.
- Each line:
  - idx: row index within this shard (0-based, consistent with embeddings row order)
  - id
  - domain: `station` or `video`
  - text_format
  - json_format
  - meta: all original fields except text_format/json_format, plus derived `main_name/name/tags/subname` (if available)

## Default paths
- Output dir default: `/mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings`

## Usage (CLI)
- Generate shard (single GPU example):
  ```bash
  python -m embedding.processor \
    --input /path/to/input.jsonl \
    --domain station \
    --output_dir /mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings \
    --gpu_id 0 --num_gpus 1
  ```
- Merge shards (after multi-GPU run):
  ```bash
  python -m embedding.processor \
    --domain station \
    --output_dir /mnt/wulinyang/to_push/gensearchrec/sid/data/embeddings \
    --num_gpus N --merge --delete_partials
  ```
