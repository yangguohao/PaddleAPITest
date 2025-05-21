# PaddleAPITest
******
## 1. 项目背景
API正确性是Paddle质量的基石，影响业务训练、推理，影响用户对Paddle的信赖。至关重要。为保障Paddle API正确性，我们开发了PaddleAPITest。

目前主要考虑的正确性问题有**3**类：
>1）API**精度**正确性；
>
>2）一些**特大**Tensor，尤其是numel超过int32上限的Tensor计算异常；
>
>3）**0-Size**（numel为0的Tensor） Tensor不支持。

PaddleAPITest主要工作思路如下：
1. 在Paddle开发Trace API机制，具体见 https://github.com/PaddlePaddle/Paddle/pull/70752 ，用于抓取API调用配置，下面是一个配置例子：
```
paddle.concat(tuple(Tensor([31376, 768],"float32"),Tensor([1, 768],"float32"),), axis=0, )
```
2. 在所有Paddle单元测试（CI）、集成测试（CE）流水线中，抓取所有Paddle API的调用配置，形成了PaddleAPITest/tester/api_config下以“api_config_CI”，“api_config_CE”开头的配置集。对以上配置集进行去重、排序、梳理得到了以“api_config_merged”开头的配置集。

3. 在 PaddleAPITest 中开发一套**引擎**，加载配置集，初始化相应Tensor，调用相应API执行前/反向测试。

4. 在 PaddleAPITest 中开发一套**转换工具**，在调用Paddle API测试的同时，等同的调用Torch API，做精度对比测试。

5. 对采集到的配置集进行shape篡改，得到了“bigtensor”、“0sizetensor”开头的配置集。

6. 通过与Torch对比，如果出现以下情况则认为Paddle API有必要确认是否正确并修复：
>a. 精度diff
>
>b. Torch正常，Paddle报错
>
>c. Torch正常，Paddle **CoreDump**或**CUDA Error**


## 2. 项目结构

目前项目结构如下所示，主要分为report和tester文件夹，report用于储存内核报错的api信息，tester用于测试配置的正确性。

- 在引擎补齐这一任务中，出现的内核报错均放置于report/fresh_report/paddle_only中。

- 在精度转换这一任务中，出现的精度报错均放置于report/fresh_report/accuracy中。

对于paddle only的测试，tester/api_config中存放测试通过（merged*）/暂未通过（merged_not_support*）的配置。

对于paddle2torch的测试，tester/api_config中存放：

- 测试通过（api_config_support2torch*）

- 存在精度问题的配置（api_config_accuracy_error*）

- 存在随机性的配置（api_config_stochastic*）

- paddle报错的配置（api_config_paddle_only_error*）

tester/api_config/config_analyzer.py是引擎补齐任务的核心代码。

tester/paddle2torch/是转换能力的核心代码。

```
├── tester
│   ├── accuracy.py
│   ├── api_config
│   ├── base.py
│   ├── paddle_cinn_vs_dygraph.py
│   ├── paddle_only.py
│   ├── paddle_to_torch
├── test_pipline
└── tools
├── engine.py
├── engineV2.py
├── engineV3.py
├── report
├── run.sh
```

tools文件夹中存放了一些实用的工具，例如move_config可以用来批量的移动配置，详见[move_config-README.md](./tools/move_config-README.md)。

## 3. 使用介绍

#### 环境配置
运行环境分为**cpu**环境与**gpu**环境，cpu和gpu上运行的结果**可能存在差异**，即存在cpu上能够正确运行，但gpu上报错的情况。因此需要根据需求正确安装环境。

下载链接：https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html

若需要本地编译paddle，可参考链接：https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html

测试CPU除了通过上述链接安装CPU的最新develop包之外，还可使用如下指令设置Paddle工作在CPU模式：
```
paddle.device.set_device("cpu")
```

#### 使用说明

**A. engine v1**

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


**B. engine v2**

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