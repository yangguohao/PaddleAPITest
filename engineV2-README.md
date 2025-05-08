# EngineV2：高性能多进程测试框架

`engineV2.py` 是为 PaddleAPITest 项目设计的高性能测试框架，支持多 GPU 并行执行，具备负载均衡、超时处理和崩溃恢复能力。相比原始的 `engine.py` 实现，它能显著提升 Paddle API 配置测试效率，加速比约为 3-5 倍。

## 功能特性

- **多 GPU 并行**：通过 `--num_gpus` 参数，支持可配置的多 GPU 执行，任务跨 GPU 动态分发。同时支持单 GPU 模式
- ~~**多 CPU 并行**：通过 `--num_cpus` 参数，支持可配置的多 CPU 执行（尚未实现）~~
- **进程级并行**：基于 Pebble 库的 ProcessPool 实现基于进程的高效任务分发，通过 `--num_workers_per_gpu` 支持单 GPU 多 worker
- **动态负载均衡**：自动将新的子进程分配给负载最轻的 GPU，确保资源最优利用
- **超时/崩溃恢复**：根据张量大小设置执行时限（90秒至3600秒梯度阈值），避免进程卡死。自动检测并重启超时或崩溃进程
- **进程安全日志**：集成 `log_writer.py` 实现无争用日志，确保日志可靠聚合
- **优雅关闭**：绑定中断信号，安全终止所有子进程
- **延迟导入**：采用类型提示和模块惰性加载，防止子进程过早初始化模块，降低内存开销
- **自动化执行**：包含 `run.sh` 脚本实现一键执行，支持 nohup 后台运行和详细监控指引

## 安装说明

1. 安装 PaddleAPITest 项目依赖项
    ```bash
    pip install paddlepaddle-gpu torch func_timeout psutil pebble
    ```
2. 克隆 PaddleAPITest 仓库并进入项目目录
   ```bash
   git clone https://github.com/PFCCLab/PaddleAPITest.git
   cd PaddleAPITest
   ```
3. 确保 `engineV2.py` 、 `log_writer.py` 和 `run.sh` 位于正确目录

## 使用指南

### 命令行参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--api_config` | str | 单条 API 配置字符串（快速测试用） |
| `--api_config_file` | str | API 配置文件路径（如`tester/api_config/api_config_temp.txt`） |
| `--paddle_only` | bool | 仅运行 Paddle 测试（默认 False） |
| `--accuracy` | bool | 启用精度测试（默认 False） |
| `--paddle_cinn` | bool | 运行 CINN vs Dygraph 对比测试（默认 False） |
| `--test_amp` | bool | 启用自动混合精度测试（默认 False） |
| `--num_gpus` | int | 使用的GPU数量（默认 0，单 GPU 模式） |
| `--num_workers_per_gpu` | int | 每个 GPU 的 worker 进程数（默认 1） |
| ~~`--num_cpus`~~ | ~~int~~ | ~~使用的CPU核心数（默认 0，尚未支持）~~ |

### 示例命令

以精度测试为例，配置文件路径为 `tester/api_config/api_config_temp.txt`，输出日志路径为 `tester/api_config/test_log`：

**多进程多 GPU 模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_temp.txt" --num_gpus=8 --num_workers_per_gpu=1 >> "tester/api_config/test_log/log.log" 2>&1
```

**单进程多 GPU 模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_temp.txt" --num_gpus=0 >> "tester/api_config/test_log/log.log" 2>&1
```

**单进程单 GPU 模式**：
```bash
export CUDA_VISIBLE_DEVICES=""
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_temp.txt" --num_gpus=0 >> "tester/api_config/test_log/log.log" 2>&1
```

**使用 run.sh 脚本**：
```bash
# chmod +x run.sh
./run.sh
```
该脚本使用默认参数（NUM_GPUS=8, NUM_WORKERS_PER_GPU=1）在后台运行程序，可在修改 `run.sh` 参数后使用

## 监控方法

执行 `run.sh` 后可通过以下方式监控：

- **GPU使用情况**：`watch -n 1 nvidia-smi`
- **日志文件**：`ls -lh tester/api_config/test_log`
- **终止进程**：`kill <PYTHON_PID>`（PID 由 run.sh 显示）
- **进程情况**：`watch -n 1 nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv`

## 核心组件

### engineV2.py

*主工作流*：
- 解析命令行参数配置执行模式（单配置或批量测试）
- 批量模式时读取 `--api_config_file` 配置，通过 `log_writer.read_log("checkpoint")` 跳过已完成测试
- 支持 `--api_config` 参数快速执行单配置测试

*多 GPU 执行*：
- 初始化 ProcessPool（最大 worker 数 = GPU 数 × 每 GPU worker 数）
- 通过 `init_worker_gpu()` 将进程绑定到GPU，使用共享的 Manager.list 跟踪分配状态
- 以 16,384 个配置为批次处理任务，优化内存和 I/O 效率

*任务执行*：
- `run_test_case()` 执行单个测试用例，根据参数选择测试类：*APITestAccuracy*、*APITestPaddleOnly*、*APITestCINNVSDygraph*
- 使用 `log_writer.write_to_log()` 记录检查点和错误

*超时处理*：
- `estimate_timeout()` 根据张量元素数量设置超时（≤1M 元素 90 秒，>100M 元素 3600 秒的梯度阈值）
- 记录并统计失败任务（超时/崩溃），自动重启进程

*清理机制*：
- `cleanup()` 终止进程池，5 秒超时等待 worker 结束
- 信号处理器（SIGINT、SIGTERM）会清理 GPU 显存（调用 torch 和 paddle 的 empty_cache）后优雅退出

### log_writer.py

- 自动创建 test_log 目录，支持安全删除
- 提供无争用的 `write_to_log()` 和 `read_log()` 方法
- 通过 `aggregate_logs()` 将各进程临时日志合并为单一文件，保持追加写入特性
- 记录已完成配置，避免重复测试

### run.sh

- 配置默认参数：NUM_GPUS=8, NUM_WORKERS_PER_GPU=1, LOG_DIR=tester/api_config/test_log
- 使用 nohup 后台运行 engineV2.py，输出重定向至 log.log
- 显示 GPU 监控、日志查看和进程终止命令

## 测试结果

- 在数千个配置上验证正确性，精度测试速度达原始引擎的 3-5 倍
- 可处理大张量（平均每 GPU 约 20GB，峰值 70GB），在多进程多 GPU、单进程多 GPU、单 GPU 模式下均表现稳定

## 注意事项

1. `estimate_timeout()` 梯度阈值粒度较粗，可进一步调整 TIMEOUT_STEPS
2. 重负载多进程情况下会产生 OOM 反复崩溃重启，可通过 nvidia-smi 监控可用显存，实现动态调整工作进程数
3. 安装 PaddlePaddle（develop） 与 PyTorch（2.6） 需确保兼容性，必须首先安装 PaddlePaddle 再安装 PyTorch

## 许可协议

遵循 PaddleAPITest 仓库的许可条款
