import pandas as pd
import json
import re

# dataname = "station_video_4B_json_format.npy_rqopq_base"
dataname = "20260303_rqkmeans.patched"

# EVAL_CSV_PATH = f"/lizhaoxuan1/llm_output/mify_video_llm_{dataname}_infer_result.csv"
EVAL_CSV_PATH = f"/lizhaoxuan1/llm_output/mify_video_llm_0305_beam_{dataname}_infer_result.csv"
# FILE_NAME = "video_llm_eval_base_24998_20260210"
FILE_NAME = "llm0306_44108_20260306"

MIFY_PATH = f"/lizhaoxuan1/mify_output/{FILE_NAME}.csv"
OUTPUT_PATH = f"/lizhaoxuan1/mify_output/parse_{FILE_NAME}.csv"

eval_df = pd.read_csv(EVAL_CSV_PATH)
df = pd.read_csv(MIFY_PATH)

def extract_query(output_str):
    if not isinstance(output_str, str):
        return None
    try:
        output_json = json.loads(output_str)
        query = output_json.get("query", "")
        return str(query).strip()
    except Exception:
        return None


eval_df["query"] = eval_df["query"].astype(str).str.strip()
df["query"] = df["Output"].apply(extract_query)
df = df.dropna(subset=["query"])

eval_df = eval_df.drop_duplicates(subset=["query"])
df = df.drop_duplicates(subset=["query"])

total_df = pd.merge(
    eval_df,
    df,
    on="query",
    how="inner",
    suffixes=("_eval", "_mify")
)

print(f"合并后总样本数: {len(total_df)}")

df_success = total_df[total_df["Status"] == "success"].copy()

print(f"success 样本数: {len(df_success)}")

def extract_score(output_str):
    if not isinstance(output_str, str):
        return None
    try:
        output_json = json.loads(output_str)
        first_res = output_json.get("first_analysis", "")
        match = re.search(r"【初步打分】：【\s*([0-4])\b", first_res)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None

df_success["score"] = df_success["Output"].apply(extract_score)
df_success = df_success.dropna(subset=["score"])

total_cnt = len(df_success)
score_4_cnt = (df_success["score"] == 4).sum()
ratio = score_4_cnt / total_cnt if total_cnt > 0 else 0

print("=" * 50)
print(f"Success 总数: {total_cnt}")
print(f"4 分数量: {score_4_cnt}")
print(f"4 分比例: {ratio:.4%}")
print("=" * 50)

df_success.to_csv(OUTPUT_PATH, index=False)
print(f"结果已保存到: {OUTPUT_PATH}")
