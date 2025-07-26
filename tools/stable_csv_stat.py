# 整理 stable*.csv 精度统计数据，产出：stable_full.csv、stable_stat.csv、stable_stat_api.csv
# @author: cangtianhuang
# @date: 2025-07-24

from pathlib import Path
import pandas as pd
import glob
from collections import defaultdict

TEST_LOG_PATH = Path("tester/api_config/test_log")
OUTPUT_PATH = TEST_LOG_PATH
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 查找所有stable*.csv文件
file_pattern = TEST_LOG_PATH / "stable*.csv"
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
    file_name = file_path.split("/")[-1]
    if file_name in ["stable_stat.csv", "stable_stat_api.csv", "stable_full.csv"]:
        continue
    try:
        df = pd.read_csv(file_path, on_bad_lines="warn")
        dfs.append(df)
        print(f"Read {len(df)} configs in {file_path}")
        config_count += len(df)
        for _, row in df.iterrows():
            api = row["API"]
            dtype = row["dtype"]
            comp = row["comp"]
            max_abs_diff = row["max_abs_diff"]
            max_rel_diff = row["max_rel_diff"]
            stats[(api, dtype, comp)]["abs_diffs"].append(max_abs_diff)
            stats[(api, dtype, comp)]["rel_diffs"].append(max_rel_diff)

            api_stats[api][dtype][comp] += 1
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
print(f"\nTotal read {len(stats)} (api, dtype, comp)s, {config_count} configs.")
if not stats:
    exit(0)

# 合并所有DataFrame并保存
merged_df = pd.concat(dfs, ignore_index=True)
numeric_cols = ["max_abs_diff", "max_rel_diff"]
merged_df = merged_df.groupby(["config", "comp"], as_index=False)[numeric_cols].mean()
# merged_df = merged_df.drop_duplicates(subset=["config", "comp"], keep="last")
merged_df = merged_df.sort_values(
    by=["API", "dtype", "config", "comp"], ignore_index=True
)
for col in numeric_cols:
    merged_df[col] = merged_df[col].apply(lambda x: f"{float(x):.6e}")
output_file = OUTPUT_PATH / "comp_full.csv"
merged_df.to_csv(output_file, index=False, na_rep="nan")

# 准备结果数据
stats_data = []
for api, dtype, comp in sorted(stats.keys()):
    values = stats[(api, dtype, comp)]
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
            "comp": comp,
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
    output_file = OUTPUT_PATH / "comp_stat.csv"
    df.to_csv(output_file, index=False, na_rep="nan")
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
        api_dtype[dtype][comp]
        for dtype in api_dtype
        for comp in api_dtype[dtype]
    )

    # 统计所有 comp 的模式
    all_comps = set()
    for dtype in api_dtype:
        all_comps.update(api_dtype[dtype].keys())
    all_comps = "/".join(sorted(all_comps))

    api_stats_data.append(
        {
            "API": api,
            "dtype": "dtypes:" + dtypes,
            "comp": f"modes:{all_comps}",
            "count": total,
            "percentage": 100.0,
        }
    )

    # 按 comp 统计
    comp_counts = {}
    for dtype in api_dtype:
        for comp in api_dtype[dtype]:
            comp_counts[comp] = comp_counts.get(comp, 0) + api_dtype[dtype][comp]

    for comp in sorted(comp_counts.keys()):
        comp_total = comp_counts[comp]
        comp_dtypes = "/".join(sorted(dtype for dtype in api_dtype if comp in api_dtype[dtype]))
        api_stats_data.append(
            {
                "API": api,
                "dtype": "dtypes:" + comp_dtypes,
                "comp": comp,
                "count": comp_total,
                "percentage": round(comp_total / total * 100, 2),
            }
        )

    # 按 dtype 和 comp 统计
    for dtype in sorted(api_dtype.keys()):
        for comp in sorted(api_dtype[dtype].keys()):
            count = api_dtype[dtype][comp]
            if count > 0:
                api_stats_data.append(
                    {
                        "API": api,
                        "dtype": dtype,
                        "mode": comp,
                        "count": count,
                        "percentage": round(count / total * 100, 2),
                    }
                )

# 转换为DataFrame并保存
if api_stats_data:
    df = pd.DataFrame(api_stats_data)
    output_file = OUTPUT_PATH / "stable_stat_api.csv"
    df.to_csv(output_file, index=False, na_rep="nan")
    print(f"\nAPI statistics saved to {output_file}")
    print("Sample of API statistics:")
    print(df.head())
else:
    print("No API statistics to process.")
