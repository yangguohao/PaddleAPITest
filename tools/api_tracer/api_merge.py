from typing import Dict, List, Optional
import pandas as pd
import yaml
from pathlib import Path


@staticmethod
def summarize_apis(
    input_path: str,
    output_path: str,
    sheet_name: Optional[str] = None,
    model_groups: Optional[Dict[str, List[str]]] = None,
    yaml_path: Optional[Dict[str, str]] = None,
):
    """
    从 XLSX 或 CSV 文件中读取 API 数据，合并并分析 API

    Args:
    - excel_path (str): 输入的 XLSX 或 CSV 文件路径
    - output_path (str): 输出的 Excel 或 TXT 文件路径（根据后缀自动判断）
    - sheet_name (str, optional): 要读取的 Excel 工作表名称. 若为 None, 则读取第一个工作表
    - model_groups (Dict[str, List[str]], optional): 模型分组字典
        例如：{'核心模型': ['model_A'], '次要模型': ['model_B']}
        如果提供, 将按此字典顺序分组并排序模型列. 如果为 None, 则无 "首次出现分组" 列
    - yaml_path (Dict[str, str], optional): API 列表的 YAML 文件路径字典, 键用于指定列名称
        例如：{'是否在总表': 'tools/api_tracer/api_list/torch_api_list.yaml'}
    """
    # Read the Excel file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"[APIMerge] Not found input file: {input_path}")

    if input_file.suffix.lower() == ".xlsx":
        df = pd.read_excel(input_file, sheet_name=sheet_name, header=None)
    elif input_file.suffix.lower() == ".csv":
        df = pd.read_csv(input_file, header=None)  # No header, as it's multi-level
        if sheet_name:
            print(f"[APIMerge] `sheet_name={sheet_name}` will be ignored for csv")
    else:
        raise ValueError(
            f"[APIMerge] Unsupported input file type: {input_file.suffix}. Only .xlsx and .csv are supported"
        )

    # Load APIs from YAML
    named_apis_set = {}
    if yaml_path is not None:
        for name, path in yaml_path.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    apis = yaml.safe_load(f)
                if not isinstance(apis, list):
                    print(f"[APIMerge] {path} should be a list, but got {type(apis)}")
                    continue
                print(f"[APIMerge] Successfully loaded {len(apis)} APIs from {path}")
                named_apis_set[name] = set(apis)
            except Exception as e:
                print(f"[APIMerge] Error loading {path}: {e}")
                continue

    # Process the dataframe to extract models and their API sets
    # Assuming structure: model names in row 0, spanning 3 columns each (merged cells appear as value in first, NaN in others)
    # Row 1: 推理 训练 总和 repeated
    # Row 2+: APIs in columns

    # Find model names and column groups
    models = []
    col_groups = []
    current_model = None
    col_start = 0
    for col in range(df.shape[1]):  # type: ignore
        val = df.iloc[0, col]  # type: ignore
        if pd.notna(val) and str(val).strip():
            if current_model:
                col_groups.append((col_start, col))
            current_model = str(val).strip()
            models.append(current_model)
            col_start = col
    if current_model:
        col_groups.append((col_start, df.shape[1]))  # type: ignore

    dup_models = []
    for model in models:
        if models.count(model) > 1:
            dup_models.append(model)
    for model_name in dup_models:
        print(f"[APIMerge] Found duplicate model: {model_name}")
    models = set(models)

    # For each model, assume 3 columns: inference, train, total
    apis_per_model = {}  # model_name -> set of APIs (merged from all three columns)
    for i, model in enumerate(models):
        start_col, end_col = col_groups[i]
        if end_col - start_col != 3:
            raise ValueError(
                f"模型 '{model}' 的列数应为3，但实际为 {end_col - start_col}。请检查文件结构。"
            )

        # 从第3行（索引为2）开始读取API
        model_api_slices = df.iloc[2:, start_col:end_col].values.flatten()
        # 清理数据：去除空值，转换为字符串，去除首尾空格，并取唯一值
        model_apis = {
            str(api).strip()
            for api in model_api_slices
            if pd.notna(api) and str(api).strip()
        }
        apis_per_model[model] = model_apis

    if not model_groups or not isinstance(model_groups, dict):
        raise ValueError("必须提供一个有效的 `model_groups` 字典用于分析。")

    # 验证模型分组的有效性
    all_grouped_models = {
        m for models_in_group in model_groups.values() for m in models_in_group
    }
    all_data_models = set(apis_per_model.keys())

    missing_models = all_grouped_models - all_data_models
    for missing_model in missing_models:
        print(f"警告: 分组中的模型 {missing_model} 在输入文件中未找到。")

    unassigned_models = all_data_models - all_grouped_models
    for unassigned_model in unassigned_models:
        print(
            f"警告: 文件中的模型 {unassigned_model} 未被分配到任何分组，但仍会显示在最终表格中。"
        )

    api_to_first_group = {}
    cumulative_apis = set()
    print("\n--- 按分组统计累计 API 数量 ---")
    for group_name, group_models in model_groups.items():
        current_group_apis = set().union(
            *(
                apis_per_model[model]
                for model in group_models
                if model in apis_per_model
            )
        )

        newly_introduced_apis = current_group_apis - cumulative_apis
        for api in newly_introduced_apis:
            api_to_first_group[api] = group_name

        cumulative_apis.update(current_group_apis)
        print(f"'{group_name}' (及之前分组) 的累计 API 总数: {len(cumulative_apis)}")

    # --- 5. 构建最终的 DataFrame ---
    all_apis = set().union(*apis_per_model.values())
    sorted_apis = sorted(list(all_apis))

    data = []
    sorted_model_names = sorted(models)

    for api in sorted_apis:
        first_group = api_to_first_group.get(api, "未分组")
        is_inlist = "是" if api in named_apis_set["是否在静态"] else "否"
        row = [api, first_group, is_inlist]
        for model_name in sorted_model_names:
            row.append("是" if api in apis_per_model.get(model_name, set()) else "否")
        data.append(row)

    columns = ["API", "首次出现分组", "是否在表中"] + sorted_model_names
    result_df = pd.DataFrame(data, columns=columns)

    # --- 6. 排序结果以便观察 ---
    # 创建一个有序的分类，用于排序“首次出现分组”列
    group_order = list(model_groups.keys()) + ["未分组"]
    result_df["首次出现分组"] = pd.Categorical(
        result_df["首次出现分组"], categories=group_order, ordered=True
    )
    result_df = result_df.sort_values(by=["首次出现分组", "API"], ignore_index=True)

    # --- 7. 输出到文件 ---
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_file_path.suffix.lower() == ".txt":
            result_df.to_csv(output_file_path, sep="\t", index=False, encoding="utf-8")
        elif output_file_path.suffix.lower() == ".xlsx":
            result_df.to_excel(output_file_path, index=False, engine="openpyxl")
        else:
            raise ValueError(
                f"不支持的输出文件类型: {output_file_path.suffix}。请使用 .xlsx 或 .txt。"
            )
        print(f"\n分析完成，总结报告已写入: {output_file_path}")
    except Exception as e:
        raise IOError(f"写入输出文件时出错: {e}")


if __name__ == "__main__":
    MODEL_GROUPS = {
        "13个": [
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2-57B-A14B",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-30B-A3B",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-4-Maverick-17B-128E",
            "llava-hf/llava-1.5-7b-hf",
            "deepseek-ai/DeepSeek-V2-Lite",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "baidu/ERNIE-4.5-0.3B-PT",
            "baidu/ERNIE-4.5-21B-A3B-PT",
            "baidu/ERNIE-4.5-VL-28B-A3B-PT",
        ],
        "19个": [
            "zai-org/GLM-4.1V-9B-Thinking",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "black-forest-labs/FLUX.1-dev",
            "ByteDance/Dolphin",
            "echo840/MonkeyOCR",
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ],
        "35个": [
            "Salesforce/blip2-opt-2.7b",
            "OpenGVLab/InternVL3-1B",
            "moonshotai/Kimi-K2-Instruct",
            "moonshotai/Kimi-VL-A3B-Instruct",
            "zai-org/GLM-4.5",
            "mistralai/Magistral-Small-2507",
            "tencent/HunyuanWorld-1",
            "MiniMaxAI/MiniMax-M1-40k",
            "state-spaces/mamba2-2.7b",
            "RWKV/RWKV7-Goose-World3-2.9B-HF",
            "deepseek-ai/Janus-Pro-1B",
            "Kwai-Keye/Keye-VL-8B-Preview",
            "XiaomiMiMo/MiMo-VL-7B-SFT",
            "ByteDance-Seed/BAGEL-7B-MoT",
            "jieliu/SD3.5M-FlowGRPO-GenEval",
            "Skywork/UniPic2-Metaquery-9B",
            "VAPO",
        ],
    }

    yaml_path = {
        "是否在静态": "tools/api_tracer/api_list/torch_api_static.yaml",
    }

    summarize_apis(
        excel_path="tools/api_tracer/Torch核心API统计(动态扫描).xlsx",
        output_path="tools/api_tracer/torch.xlsx",
        sheet_name="各模型API",
        model_groups=MODEL_GROUPS,
        yaml_path=yaml_path,
    )
