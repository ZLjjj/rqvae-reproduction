import json
import random
from tqdm import tqdm

dataname = "rqvae_opq_20260208_station_video_text_format_4B"

ALIGN_TRAIN_JSONL_PATH = f"/lizhaoxuan1/train_data/align_train_station_0.95_video_0.98_rqvae_opq_20260208_station_video_text_format_4B.jsonl"
OUTPUT_TRAIN_IP_JSONL_PATH = f"/lizhaoxuan1/train_data/ext_align_video_ip_{dataname}.jsonl"
OUTPUT_TRAIN_STAFF_JSONL_PATH = f"/lizhaoxuan1/train_data/ext_align_video_staff_{dataname}.jsonl"

TRAIN_FILE_NAME = f"align_train_station_video_{dataname}_merged"
OUTPUT_TRAIN_JSONL_PATH = f"/lizhaoxuan1/train_data/{TRAIN_FILE_NAME}.jsonl"

DATASET_INFO_PATH = f"/lizhaoxuan1/gensearchrec/llama-factory/data/dataset_info.json"

RANDOM_SEED = 2026

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skip bad json: {path} line={line_num} err={e}")
    return records


def dump_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def merge_and_shuffle_trainset():
    random.seed(RANDOM_SEED)

    print("Loading align train data...")
    align_records = load_jsonl(ALIGN_TRAIN_JSONL_PATH)
    print(f"  align train samples: {len(align_records)}")

    print("Loading IP SFT data...")
    ip_records = load_jsonl(OUTPUT_TRAIN_IP_JSONL_PATH)
    print(f"  ip samples: {len(ip_records)}")

    print("Loading staff SFT data...")
    staff_records = load_jsonl(OUTPUT_TRAIN_STAFF_JSONL_PATH)
    print(f"  staff samples: {len(staff_records)}")

    # 增大ip比例
    all_records = align_records + ip_records + ip_records + staff_records
    print(f"Total samples before shuffle: {len(all_records)}")

    print("Shuffling...")
    random.shuffle(all_records)

    print("Writing merged dataset...")
    dump_jsonl(all_records, OUTPUT_TRAIN_JSONL_PATH)

    print("Done!")
    print(f"Final dataset saved to: {OUTPUT_TRAIN_JSONL_PATH}")
    print(f"Final sample count: {len(all_records)}")

def update_dataset_info(key, file_path, config_path):
    item = {
        "file_name": file_path,
        "formatting": "sharegpt",
        "columns": {
        "messages": "messages"
        },
        "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
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
    merge_and_shuffle_trainset()
    update_dataset_info(TRAIN_FILE_NAME, OUTPUT_TRAIN_JSONL_PATH, DATASET_INFO_PATH)
