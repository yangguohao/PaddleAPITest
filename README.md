# PaddleAPITest

******

## 1. 项目背景

正确性是Paddle质量的基石，影响业务训练、推理，影响用户对Paddle的信赖。至关重要。
如何主动发现Paddle存在的质量问题，给予修复。而不是被动等用户反馈后，再修复。是一个质量上的难题。
API是Paddle的“窗口”，PaddleAPITest通过整理API级的300余万条case（本项目称之为配置 / api config），使用case驱动Paddle不同内核机制执行API，可成为Paddle内核机制、算子正确性的“扫描仪”。

PaddleAPITest主要工作思路如下：

1. 在Paddle开发Trace API机制，具体见 https://github.com/PaddlePaddle/Paddle/pull/70752 ，用于抓取API调用配置，下面是一个配置例子：

```
paddle.concat(tuple(Tensor([31376, 768],"float32"),Tensor([1, 768],"float32"),), axis=0, )
```

2. 在所有Paddle单元测试（CI）、集成测试（CE）流水线中，抓取所有Paddle API的调用配置，形成了PaddleAPITest/tester/api_config下以 "CI_CE_config" 的配置集。对以上配置集进行去重、排序并对测试结果进行梳理得到了 api_config 下各个目录的配置集。

3. 在 PaddleAPITest 中开发一套**引擎**，加载配置集，初始化相应Tensor，调用相应API执行前/反向测试。

4. 对采集到的配置集进行shape篡改，得到了 "big_tensor"、"0_size" 开头的配置集。

5. 对于精度正确性，在 PaddleAPITest 中开发一套**转换工具**，在调用Paddle API测试的同时，等同的调用Torch API，做精度对比测试。

6. 对于内核测试，可通过继承APITestBase，开发针对某内核的测试引擎。被测对象可以是：Kernel精度、Kernel性能、Kernel显存、动态图、静态图、动转静、组合算子、CINN、Paddle2ONNX、GPU、CPU、XPU、NPU、OneDNN、TRT等等。

## 2. 项目结构

```python
├── tester
│   ├── accuracy.py
│   ├── api_config
│   ├── base.py
│   ├── paddle_cinn_vs_dygraph.py
│   ├── paddle_only.py
│   └── paddle_to_torch
├── test_pipline
├── tools
├── engine.py
├── engineV2.py
├── engineV3.py
├── report
└── run-example.sh
```

目前项目结构主要分为 report 和 tester 文件夹，report 用于储存内核报错的 api 信息，tester 用于测试配置的正确性和存放配置测试结果。

engineV2.py 及配合的 run-example.sh 是目前运行本项目的主要工具，engineV3.py 目前由百度内部开发测试使用，engine.py 是最早的引擎，相较 engineV2.py 吞吐量低，在少量配置时可使用。

1. report 介绍
   - 0size_tensor_cpu 存放进行在 cpu 上进行精度测试/引擎解析能力测试（accuracy / paddle_only）结果。
   - 0size_tensor_gpu 存放进行在 gpu 上进行 accuracy / paddle_only 测试结果。
   - big_tensor_gpu 存放大形状张量（big tensor）在 gpu/cpu 上进行 accuracy / paddle_only 测试结果。
   - ci_ce_cpu 存放 CI/CE 流水线抓取的配置在 cpu 上进行 accuracy / paddle_only 测试结果。
   - ci_ce_gpu 存放 CI/CE 流水线抓取的配置在 gpu 上进行 accuracy / paddle_only 测试结果。
   - cinn 存放 paddle 静态编译器与动态图方式进行精度对比测试结果。
   - fresh_report 存放引擎补齐（paddle_only）和精度转换（accuracy）两个任务中，出现的内核报错或者精度报错。


2. tester 介绍

   * api_config 目录存放配置目录的管理情况和相关脚本工具，各配置目录下的文本命名语义一致，参考 5_accuracy 中的 txt 命名含义：
     * 1_not_support 目录下的配置为 paddle 不支持的配置，例如数据类型对应的算子未实现。
     * 2_paddle_only_random 为 PaddleAPITest 引擎支持对配置的解析（--paddle_only=True），但是配置输出随机，也就无法继续进行精度测试的配置。random_creation.txt 为创建随机数的配置，random_calculation.txt 为具有随机行为的函数。
     * 3_paddle_only 支持引擎解析（本项目称为 merged），但是无法进行精度测试（也称为 merged_not_support）的配置。
     * 4_paddle_only_amp 在混合精度模式下支持引擎解析的配置（--paddle_only=True --test_amp=True）。
     * 5_accuracy 支持精度测试的配置，不同的命名对应不同的测试结果：
       * accuracy_*.txt 为测试通过。
       * accuracy_cpu_error.txt 为 **在 cpu 上运行精度测试不通过（--test_cpu=True）** 的配置集。
       * accuracy_cpu_kernel.txt 为 **paddle 内核抛出错误** 的配置集。
       * accuracy_gpu_error_dtype_diff.txt 为回归测试中发现精度错误，但是检查精度时 **不强制对齐 dtype 后能够通过** 的配置（base.py 中的 not_check_dtype 列表中的 api）。
       * accuracy_gpu_error_grads_diff.txt 为 **paddle 和 torch api 反向梯度结果不同，无法进行比较，输出 [not compare]** 的配置集。
       * accuracy_gpu_error_uncertain.txt 为 **不确定是因为 PaddleAPITest 引擎转化能力 还是 Paddle 内核实现** 导致的 accuracy error 的配置集。
       * accuracy_cpu_error.txt 为 **在 cpu 上运行精度测试不通过（--test_cpu=True）** 的配置集。
     * 6_accuracy_amp 类似 5_accuracy，只是需要混合精度运行测试（--test_amp=True）的配置集。
     * 7_0_size 张量形状含有 0（0-size）的配置集。
     * 8_big_tensor 对配置的张量形状基于 to_big_size_config.py 进行篡改的配置集。
     * 9_getset_item 为测试 paddle.Tensor.\_\_getitem__ 和 paddle.Tensor.\_\_getitem__ 使用的配置集。
     * big_and_0size 为含有大形状张量和 0-size 张量使用的配置集。
     * CI_CE_config 为 CI，CE 抓取的配置集。
     * 脚本工具
       * config_analyzer.py 是引擎对配置解析并针对 api 初始化合适张量的代码（引擎补齐任务产物）
       * log_writer.py 是 engine 写入日志的工具
       * to_0_size*.py 是篡改为 0-size 配置的工具
       * to_big_size\*.py 是篡改为大形状张量的配置的工具。 

   * tester/paddle2torch/是转换能力的核心代码。介绍详见 [4.paddle2torch转换](#4-paddle2torch转换)

3. tools文件夹中存放了一些实用的工具，例如move_config可以用来批量的移动配置，详见[move_config-README.md](./tools/move_config-README.md)。

## 3. 使用介绍

### 环境配置

运行环境分为**cpu**环境与**gpu**环境，cpu和gpu上运行的结果**可能存在差异**，即存在cpu上能够正确运行，但gpu上报错的情况。因此需要根据需求正确安装环境。

[PaddlePaddle 安装链接](https://www.paddlepaddle.org.cn/install/quick)

若需要本地编译paddle，可参考链接：https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html

测试CPU除了通过上述链接安装CPU的最新develop包之外，还可使用如下指令设置Paddle工作在CPU模式：

```
paddle.device.set_device("cpu")
```

### 使用说明

#### A. engine v1

所有测试前，**必须创建**一个目录：PaddleAPITest/tester/api_config/test_log/，用于存放测试所产生的测试结果和checkpoint。

PaddleAPITest目前支持paddle_only、accuracy、paddle_cinn三种测试：

>paddle_only，用于单纯把配置在Paddle动态图跑一遍，验证PaddleAPITest 引擎**是否支持**该配置。
>
>accuracy，用于将Paddle API的前反向与**Torch**的前反向做精度对比测试。
>
>paddle_cinn，用于Paddle动态图与Paddle静态图编译器做精度对比测试。

当测试**单个配置**时，可使用下面的代码，--api_config中输入待测试的配置内容：

仅测试paddle**是否支持**：

```
python engine.py --paddle_only=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'
```

测试输出**是否准确**：

```
python engine.py --accuracy=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'
```

动态图和静态图测试：

```
python engine.py --paddle_cinn=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'
```

**值得注意**的是配置txt中统一使用双引号"，因此建议--api_config=''使用单引号，或在配置中手动添加转义斜杠\

当需要测试的配置数目较多时，手动单次输入将**非常低效**，这种情况下可以使用如下所示的**批量测试**指令，将配置保存在一个txt中，并将指令中的路径设置为txt的路径即可：

```
python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --accuracy=True > tester/api_config/test_log/log.log 2>&1

python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_only=True > tester/api_config/test_log/log.log 2>&1

python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_cinn=True > tester/api_config/test_log/log.log 2>&1
```

当测试配置中有**精度不统一**的情况，需要精度转换时，直接运行测试可能会报错，可加入--test_amp=True


#### B. engine v2

`engineV2.py` 是为 PaddleAPITest 项目设计的高性能测试框架，支持多 GPU 并行执行，具备负载均衡、超时处理和崩溃恢复能力。相比原始的 `engine.py` 实现，它能显著提升 Paddle API 配置测试效率，加速比约为 5-10 倍。

功能特性:

- **多 GPU 并行**：拥有灵活的 *gpus 相关参数*，可配置多 GPU 并行测试，支持任务跨 GPU 动态分发
- **进程级并行**：基于 Pebble 库的 ProcessPool 进程池实现，支持进程的高效并行，每张 GPU 可拥有多个 worker
- **动态负载均衡**：新的子进程自动分配至负载最轻 GPU，计算资源最优利用
- **超时/崩溃恢复**：由张量大小推断执行时限（梯度阈值），主动杀死 coredump 进程，自动检测并重启死亡进程

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

该脚本使用参数：NUM_GPUS=-1, NUM_WORKERS_PER_GPU=-1，在后台运行程序，可在修改 `run.sh` 参数后使用

其说明文档详见 [engineV2.md](./engineV2.md)


## 4. paddle2torch转换

Paddle2Torch 是一个专注于将 PaddlePaddle API 转换为 PyTorch 对应实现的知识工具库，属于 [PaddleAPITest](https://github.com/PFCCLab/PaddleAPITest) 项目的核心组成模块。本模块通过解析 PaddlePaddle API 调用，使用预定义的转换规则与动态代码生成，实现从 PaddlePaddle 到 PyTorch 的自动转换。转换过程将确保代码的语义一致性。

本模块具有精简强悍的架构，仅由三个组件构成：

- *转换引擎 converter.py*
- *转换配置 mapping.json*
- *转换规则 rules.py*

代码已完全进行解耦，可以非常容易地迁移至其他代码中。本模块通过 **转换配置** 与 **转换规则** 管理 API 映射关系，因此支持开发者灵活扩展新的 API 转换能力。

本模块的典型应用场景包括：模型迁移、跨框架验证、混合编程等，可为深度学习开发者提供跨框架的互操作性解决方案。

现在转换工具已基本完成对PaddleAPI的转换。

其说明文档详见 [paddle2torch.md](./tester/paddle_to_torch/paddle2torch.md)