# 整理 tol_*.csv 精度统计数据，产出：tol_full.csv、tol_stat.csv、tol_stat_api.csv
# @author: cangtianhuang
# @date: 2025-06-20

from pathlib import Path
import pandas as pd
import glob
from collections import defaultdict

TEST_LOG_PATH = Path("tester/api_config/test_log")
OUTPUT_PATH = TEST_LOG_PATH
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 查找所有tol_*.csv文件
file_pattern = TEST_LOG_PATH / "tol*.csv"
file_list = glob.glob(str(file_pattern))
file_list.sort()
if not file_list:
    print(f"No files found matching pattern {file_pattern}")
    exit(0)

# 读取并处理每个文件
dfs = []
stats = defaultdict(lambda: defaultdict(list))
api_dtype_counts = defaultdict(lambda: defaultdict(int))
config_count = 0
for file_path in file_list:
    if (
        file_path.split("/")[-1] == "tol_stat.csv"
        or file_path.split("/")[-1] == "tol_stat_api.csv"
        or file_path.split("/")[-1] == "tol_full.csv"
    ):
        continue
    try:
        df = pd.read_csv(file_path)
        df = df.drop_duplicates(subset=["config"], keep="last")
        dfs.append(df)
        print(f"Read {len(df)} configs in {file_path}")
        config_count += len(df)
        for _, row in df.iterrows():
            api = row["API"]
            dtype = row["dtype"]
            max_abs_diff = row["max_abs_diff"]
            max_rel_diff = row["max_rel_diff"]
            stats[(api, dtype)]["abs_diffs"].append(max_abs_diff)
            stats[(api, dtype)]["rel_diffs"].append(max_rel_diff)

            api_dtype_counts[api][dtype] += 1
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
print(f"\nTotal read {len(stats)} (api, dtype)s, {config_count} configs.")
if not stats:
    exit(0)

# 合并所有DataFrame并保存
merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df.drop_duplicates(subset=["config"], keep="last")
merged_df = merged_df.sort_values(by=["API", "dtype", "config"]).reset_index(drop=True)
numeric_cols = ["max_abs_diff", "max_rel_diff"]
for col in numeric_cols:
    merged_df[col] = merged_df[col].apply(lambda x: f"{float(x):.6e}")
output_file = OUTPUT_PATH / "tol_full.csv"
merged_df.to_csv(output_file, index=False)

# 准备结果数据
result_data = []
for api, dtype in sorted(stats.keys()):
    values = stats[(api, dtype)]
    abs_diffs = values["abs_diffs"]
    rel_diffs = values["rel_diffs"]

    abs_min = min(abs_diffs)
    abs_max = max(abs_diffs)
    abs_mean = sum(abs_diffs) / len(abs_diffs)
    rel_min = min(rel_diffs)
    rel_max = max(rel_diffs)
    rel_mean = sum(rel_diffs) / len(rel_diffs)
    count = len(abs_diffs)

    result_data.append(
        {
            "API": api,
            "dtype": dtype,
            "abs_min": "{:.6e}".format(abs_min),
            "abs_max": "{:.6e}".format(abs_max),
            "abs_mean": "{:.6e}".format(abs_mean),
            "rel_min": "{:.6e}".format(rel_min),
            "rel_max": "{:.6e}".format(rel_max),
            "rel_mean": "{:.6e}".format(rel_mean),
            "count": count,
        }
    )

# 转换为DataFrame并保存
if result_data:
    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values(by=["API", "dtype"])

    output_file = OUTPUT_PATH / "tol_stat.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nStatistics saved to {output_file}")
    print("Sample of the results:")
    print(result_df.head())
else:
    print("No data to process.")

# 准备统计数据
api_stats = []
for api in sorted(api_dtype_counts.keys()):
    dtype_counts = api_dtype_counts[api]
    total = sum(dtype_counts.values())

    api_stats.append(
        {
            "API": api,
            "dtype": "TOTAL",
            "count": total,
            "percentage": 100.0,
        }
    )

    for dtype in sorted(dtype_counts.keys()):
        api_stats.append(
            {
                "API": api,
                "dtype": dtype,
                "count": dtype_counts[dtype],
                "percentage": round(dtype_counts[dtype] / total * 100, 2),
            }
        )

# 转换为DataFrame并保存
if api_stats:
    api_df = pd.DataFrame(api_stats)
    api_df = api_df.sort_values(by=["API", "dtype"])

    api_output_file = OUTPUT_PATH / "tol_stat_api.csv"
    api_df.to_csv(api_output_file, index=False)
    print(f"\nAPI statistics saved to {api_output_file}")
    print("Sample of API statistics:")
    print(api_df.head())
else:
    print("No API statistics to process.")
