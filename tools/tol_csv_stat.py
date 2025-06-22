# 整理 tol_*.csv 精度统计数据，产出：tol_full.csv、tol_stat.csv、tol_stat_api.csv
# @author: cangtianhuang
# @date: 2025-06-21

from pathlib import Path
import pandas as pd
import glob
from collections import defaultdict

TEST_LOG_PATH = Path("tester/api_config/test_log")
OUTPUT_PATH = TEST_LOG_PATH
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 查找所有tol*.csv文件
file_pattern = TEST_LOG_PATH / "tol*.csv"
file_list = glob.glob(str(file_pattern))
file_list.sort()
if not file_list:
    print(f"No files found matching pattern {file_pattern}")
    exit(0)

# 读取并处理每个文件
dfs = []
stats = defaultdict(lambda: defaultdict(list))
api_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
config_count = 0
for file_path in file_list:
    if (
        file_path.split("/")[-1] == "tol_stat.csv"
        or file_path.split("/")[-1] == "tol_stat_api.csv"
        or file_path.split("/")[-1] == "tol_full.csv"
    ):
        continue
    try:
        df = pd.read_csv(file_path, on_bad_lines="warn")
        dfs.append(df)
        print(f"Read {len(df)} configs in {file_path}")
        config_count += len(df)
        for _, row in df.iterrows():
            api = row["API"]
            dtype = row["dtype"]
            mode = row["mode"]
            max_abs_diff = row["max_abs_diff"]
            max_rel_diff = row["max_rel_diff"]
            stats[(api, dtype, mode)]["abs_diffs"].append(max_abs_diff)
            stats[(api, dtype, mode)]["rel_diffs"].append(max_rel_diff)

            api_stats[api][dtype][mode] += 1
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
print(f"\nTotal read {len(stats)} (api, dtype, mode)s, {config_count} configs.")
if not stats:
    exit(0)

# 合并所有DataFrame并保存
merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df.drop_duplicates(subset=["config", "mode"], keep="last")
merged_df = merged_df.sort_values(
    by=["API", "dtype", "config", "mode"], ignore_index=True
)
numeric_cols = ["max_abs_diff", "max_rel_diff"]
for col in numeric_cols:
    merged_df[col] = merged_df[col].apply(lambda x: f"{float(x):.6e}")
output_file = OUTPUT_PATH / "tol_full.csv"
merged_df.to_csv(output_file, index=False)

# 准备结果数据
stats_data = []
for api, dtype, mode in sorted(stats.keys()):
    values = stats[(api, dtype, mode)]
    abs_diffs = values["abs_diffs"]
    rel_diffs = values["rel_diffs"]

    abs_min = min(abs_diffs)
    abs_max = max(abs_diffs)
    abs_mean = sum(abs_diffs) / len(abs_diffs)
    rel_min = min(rel_diffs)
    rel_max = max(rel_diffs)
    rel_mean = sum(rel_diffs) / len(rel_diffs)
    count = len(abs_diffs)

    stats_data.append(
        {
            "API": api,
            "dtype": dtype,
            "mode": mode,
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
if stats_data:
    df = pd.DataFrame(stats_data)
    output_file = OUTPUT_PATH / "tol_stat.csv"
    df.to_csv(output_file, index=False)
    print(f"\nStatistics saved to {output_file}")
    print("Sample of the results:")
    print(df.head())
else:
    print("No data to process.")

# 准备统计数据
api_stats_data = []
for api in sorted(api_stats.keys()):
    api_dtype = api_stats[api]
    dtypes = "/".join(sorted(api_dtype.keys()))
    total = sum(
        api_dtype[dtype]["forward"] + api_dtype[dtype]["backward"]
        for dtype in api_dtype
    )

    api_stats_data.append(
        {
            "API": api,
            "dtype": "dtypes:" + dtypes,
            "mode": "modes:forward/backward",
            "count": total,
            "percentage": 100.0,
        }
    )

    forward_dtypes = []
    forward_total = 0
    backward_dtypes = []
    backward_total = 0
    for dtype, modes in api_dtype.items():
        if "forward" in modes:
            forward_dtypes.append(dtype)
            forward_total += modes["forward"]
        if "backward" in modes:
            backward_dtypes.append(dtype)
            backward_total += modes["backward"]
    forward_dtypes = "/".join(sorted(forward_dtypes))
    backward_dtypes = "/".join(sorted(backward_dtypes))

    api_stats_data.append(
        {
            "API": api,
            "dtype": "dtypes:" + forward_dtypes,
            "mode": "forward",
            "count": forward_total,
            "percentage": round(forward_total / total * 100, 2),
        }
    )
    api_stats_data.append(
        {
            "API": api,
            "dtype": "dtypes:" + backward_dtypes,
            "mode": "backward",
            "count": backward_total,
            "percentage": round(backward_total / total * 100, 2),
        }
    )

    for dtype in sorted(api_dtype.keys()):
        for mode in ["forward", "backward"]:
            count = api_dtype[dtype][mode]
            if count > 0:
                api_stats_data.append(
                    {
                        "API": api,
                        "dtype": dtype,
                        "mode": mode,
                        "count": count,
                        "percentage": round(count / total * 100, 2),
                    }
                )

# 转换为DataFrame并保存
if api_stats_data:
    df = pd.DataFrame(api_stats_data)
    output_file = OUTPUT_PATH / "tol_stat_api.csv"
    df.to_csv(output_file, index=False)
    print(f"\nAPI statistics saved to {output_file}")
    print("Sample of API statistics:")
    print(df.head())
else:
    print("No API statistics to process.")
