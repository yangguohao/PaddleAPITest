import os
import datetime
from collections import defaultdict

def filter_files_inplace(
    error_file_path,
    source_file_paths,
    filtered_error_output_path, # 清理（去重去空）后的错误配置文件的保存路径
    deleted_log_path="filter_log.txt", # 日志文件路径
    user_info=None
):
    """
    根据错误文件过滤源文件（直接修改源文件），过滤错误文件本身，
    并记录删除/丢弃的信息到日志文件。

    Args:
        error_file_path (str): 原始错误配置文件的路径。
        source_file_paths (list): 需要被过滤并直接修改的源文件路径列表。
        filtered_error_output_path (str): 清理后的错误配置文件的保存路径。
        deleted_log_path (str): 记录删除/丢弃信息的日志文件路径。
        user_info (dict): 包含用户信息和时间的字典。
    """

    # --- 警告 ---
    print("*" * 60)
    print("警告：此脚本将直接修改源文件！")
    print("操作不可逆，请确保您已备份原始文件。")
    print("*" * 60)
    # 可以取消下面的注释来增加一个确认步骤
    # confirmation = input("输入 'yes' 以确认继续执行直接修改操作: ")
    # if confirmation.lower() != 'yes':
    #     print("操作已取消。")
    #     return

    unique_error_lines = set()
    original_lines_processed = set()
    duplicate_lines_details = defaultdict(list)
    empty_or_whitespace_lines = []
    raw_error_line_count = 0
    unique_lines_content = []

    # --- 1. 读取并处理错误配置文件 (与之前版本相同) ---
    try:
        print(f"--- 开始读取并处理错误配置文件 ---")
        print(f"原始文件路径: {error_file_path}")
        with open(error_file_path, 'r', encoding='utf-8') as f_err:
             for line_num, line in enumerate(f_err, 1):
                raw_error_line_count += 1
                original_line = line
                stripped_line = line.strip()
                if not stripped_line:
                    empty_or_whitespace_lines.append(f"[行 {line_num}] {original_line.rstrip()}")
                elif stripped_line in original_lines_processed:
                    if stripped_line not in duplicate_lines_details:
                         first_occurrence_line = "[未找到首次出现的原始行]"
                         for idx, (orig_l, strip_l) in enumerate(unique_lines_content):
                              if strip_l == stripped_line:
                                   first_occurrence_line = f"[首次出现于行 {idx+1}] {orig_l.rstrip()}"
                                   break
                         duplicate_lines_details[stripped_line].append(first_occurrence_line)
                    duplicate_lines_details[stripped_line].append(f"[重复于行 {line_num}] {original_line.rstrip()}")
                else:
                    unique_error_lines.add(stripped_line)
                    original_lines_processed.add(stripped_line)
                    unique_lines_content.append((original_line, stripped_line))

        print(f"从错误配置文件读取了 {raw_error_line_count} 行。")
        print(f"识别出 {len(unique_error_lines)} 条唯一的、非空的错误配置。")
        num_empty = len(empty_or_whitespace_lines)
        num_duplicates = sum(len(v)-1 for v in duplicate_lines_details.values())
        print(f"识别出 {num_empty} 行空行或仅含空白，{num_duplicates} 次重复配置。")

        print(f"正在写入清理后的错误配置文件到: {filtered_error_output_path}")
        with open(filtered_error_output_path, 'w', encoding='utf-8') as f_out_err:
            for original_line, _ in unique_lines_content:
                f_out_err.write(original_line)
        print(f"清理后的错误配置文件写入完成。")
        print(f"--- 错误配置文件处理完毕 ---")

    except FileNotFoundError:
        print(f"错误：原始错误配置文件未找到于 {error_file_path}")
        print("处理中止。")
        return
    except Exception as e:
        print(f"处理错误配置文件 {error_file_path} 时发生错误: {e}")
        print("处理中止。")
        return

    # --- 2. 准备并写入日志文件 (与之前版本相同) ---
    total_lines_skipped_from_sources = 0
    processed_source_file_count = 0

    try:
        print(f"\n--- 准备过滤日志文件 ---")
        print(f"日志文件路径: {deleted_log_path}")
        with open(deleted_log_path, 'w', encoding='utf-8') as f_log:
            f_log.write("--- 配置过滤日志 ---\n")
            if user_info and 'datetime' in user_info:
                f_log.write(f"执行时间 (UTC): {user_info['datetime']}\n")
            else:
                 now_utc = datetime.datetime.now(datetime.timezone.utc)
                 f_log.write(f"记录生成时间 (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if user_info and 'user' in user_info:
                f_log.write(f"执行用户: {user_info['user']}\n")
            f_log.write(f"原始错误文件: {error_file_path}\n")
            f_log.write(f"清理后错误文件: {filtered_error_output_path}\n")
            f_log.write("操作模式: 直接修改源文件\n") # 标明操作模式
            f_log.write("--------------------------\n\n")

            f_log.write(f"--- 错误文件 '{os.path.basename(error_file_path)}' 处理详情 ---\n")
            # (日志记录错误文件详情部分不变)
            if not empty_or_whitespace_lines and not duplicate_lines_details:
                 f_log.write("错误文件中所有行都是唯一的、非空的。\n")
            else:
                if empty_or_whitespace_lines:
                    f_log.write(f"\n检测到 {len(empty_or_whitespace_lines)} 行空行或仅包含空白 (已丢弃):\n")
                    for line_info in empty_or_whitespace_lines: f_log.write(f"  - {line_info}\n")
                else: f_log.write("\n未检测到空行或仅包含空白的行。\n")
                if duplicate_lines_details:
                    f_log.write(f"\n检测到 {len(duplicate_lines_details)} 种内容的配置存在重复 (仅保留首次出现):\n")
                    for stripped, originals in duplicate_lines_details.items():
                        f_log.write(f"\n  内容: {stripped}\n")
                        for line_info in originals: f_log.write(f"    - {line_info}\n")
                else: f_log.write("\n未检测到重复的配置内容。\n")
            f_log.write("--------------------------------------------------\n\n")

            f_log.write(f"--- 源文件过滤详情 (直接修改，记录删除的行) ---\n")
            if not source_file_paths:
                 f_log.write("没有提供需要过滤的源文件。\n")

            # --- 3. 处理源文件（直接修改）并记录删除的行 ---
            print(f"\n--- 开始处理并直接修改源文件 ---")
            for source_path in source_file_paths:
                if not os.path.exists(source_path):
                    print(f"\n警告：源文件未找到，跳过: {source_path}")
                    f_log.write(f"\n警告：文件未找到，跳过 - {source_path}\n")
                    continue

                print(f"\n正在处理文件 (直接修改): {source_path}")
                lines_to_keep = [] # 存储需要保留的行
                lines_skipped_in_file = 0 # 当前文件删除的行数
                current_raw_line_count = 0
                file_has_deletions = False

                try:
                    # --- 核心改动：先读入内存 ---
                    with open(source_path, 'r', encoding='utf-8') as f_in:
                        original_lines = f_in.readlines() # 读取所有行到列表
                        current_raw_line_count = len(original_lines)

                    # --- 遍历内存中的行进行判断 ---
                    for line_num, line in enumerate(original_lines, 1):
                        stripped_line = line.strip()
                        if stripped_line in unique_error_lines:
                            # 行需要删除
                            lines_skipped_in_file += 1
                            if not file_has_deletions:
                                f_log.write(f"\n[从文件 '{os.path.basename(source_path)}' 中删除]\n")
                                file_has_deletions = True
                            f_log.write(f"  [行 {line_num}] {line.rstrip()}\n")
                        else:
                            # 行需要保留
                            lines_to_keep.append(line) # 保留原始行（带换行符）

                    # --- 核心改动：写回原文件 ---
                    if lines_skipped_in_file > 0: # 只有发生删除时才需要重写
                         print(f"  检测到 {lines_skipped_in_file} 行需要删除，正在覆盖原文件...")
                         with open(source_path, 'w', encoding='utf-8') as f_out:
                              f_out.writelines(lines_to_keep) # 将保留的行写回
                         print(f"  文件已覆盖。")
                    else:
                         print(f"  文件无需修改。")


                    lines_written = len(lines_to_keep)
                    print(f"  原始行数: {current_raw_line_count}")
                    print(f"  保留行数: {lines_written}")
                    print(f"  删除行数: {lines_skipped_in_file}")
                    if lines_written + lines_skipped_in_file != current_raw_line_count:
                         print(f"  警告：计数校验失败，请检查逻辑！")
                         f_log.write(f"警告：文件 '{os.path.basename(source_path)}' 计数校验失败！\n")

                    if lines_skipped_in_file > 0:
                         total_lines_skipped_from_sources += lines_skipped_in_file
                    processed_source_file_count += 1

                except Exception as e:
                    error_msg = f"处理文件 {source_path} 时发生错误: {e}"
                    print(error_msg)
                    f_log.write(f"\n!!! {error_msg} !!!\n\n") # 记录错误到日志

        # 日志文件 'with' 块结束
        print(f"\n过滤日志已写入: {deleted_log_path}")

    except Exception as e:
         print(f"错误：无法写入过滤日志文件 {deleted_log_path}: {e}")

    # --- 4. 打印总结信息 ---
    print(f"\n--- 处理总结 ---")
    print(f"成功处理了 {processed_source_file_count} 个源文件。")
    print(f"从源文件中总共删除（跳过）了 {total_lines_skipped_from_sources} 行。")
    print(f"源文件已被直接修改。") # 更新提示
    print(f"清理后的错误配置文件已保存为: {filtered_error_output_path}")
    print(f"详细的过滤日志已保存到: {deleted_log_path}")

# --- 配置区 ---
original_error_file = "api_config_paddleonly_error.txt"
filtered_error_file = "api_config_paddleonly_error_unique.txt"
log_file = "filter_log.txt"

source_files_to_process = []
for i in range(1, 14):
    source_files_to_process.append(f"api_config_merged_{i}.txt")
source_files_to_process.append("api_config_merged_amp.txt")

# 使用您提供的最新用户信息和时间
user_context = {
    'user': 'yuwu46',
    'datetime': '2025-04-25 04:00:15' # 更新时间戳
}

# --- 执行过滤 (直接修改模式) ---
if __name__ == "__main__":
    filter_files_inplace( # 注意函数名已更改
        error_file_path=original_error_file,
        source_file_paths=source_files_to_process,
        filtered_error_output_path=filtered_error_file,
        deleted_log_path=log_file,
        user_info=user_context
    )
    print("\n过滤处理完成。源文件已被修改。")