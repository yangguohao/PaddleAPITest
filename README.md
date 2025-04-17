# PaddleAPITest
******
## 1. 项目背景
公司内部业务，或者Paddle Issue，偶尔会反馈Paddle API正确性的问题，经过梳理大致有**3**类问题：
>1）API**精度**不正确；
>
>2）一些**特大**Tensor，尤其是numel超过int32上限的Tensor计算异常；
>
>3）**0-Size**（numel为0的Tensor） Tensor不支持。
API正确性是Paddle质量的基石，影响业务训练、推理，影响用户对Paddle的信赖。至关重要。我们需要统一排查问题API、问题case，统一修复。

为此，我们开始着手发了PaddleAPITest用于排查问题API、问题case。主要工作思路如下：
1. 在Paddle开发Trace API机制，用于抓取API调用配置，下面是一个例子：
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

在引擎补齐这一任务中，出现的内核报错均放置于fresh_report中。

tester/api_config中存放测试通过（merged*）/暂未通过（merged_not_support*）的配置。

config_analyzer.py是引擎补齐任务的核心代码。

```
├── engine.py
├── __init__.py
├── README.md
├── report
│   ├── 0size_tensor
│   │   ├── compare_with_torch
│   │   └── not_compare_with_torch
│   ├── big_tensor
│   │   ├── compare_with_torch
│   │   └── not_compare_with_torch
│   ├── ci_ce
│   │   ├── error_api.txt
│   │   ├── error_config.txt
│   │   ├── error_log.log
│   │   └── test_code.py
│   └── fresh_report

└── tester
    ├── accuracy.py
    ├── api_config
    │   ├── 0sizetensor_accuracy.txt
    │   ├── 0sizetensor_paddleonly.txt
    │   ├── api_config_0_size_coredump.txt
    │   ├── api_config_0_size.txt
    │   ├── api_config_big_tensor.txt
    │   ├── api_config_CE_Benchmark1.txt
    │   ├── api_config_CE_Benchmark2.txt
    │   ├── api_config_CE_PaddleMIX.txt
    │   ├── api_config_CE_PaddleNLP_gpt3.txt
    │   ├── api_config_CE_PaddleNLP_llm.txt
    │   ├── api_config_CE_PaddleNLP_unittest.txt
    │   ├── api_config_CE_PaddleX.txt
    │   ├── api_config_CI_CE_Framework.txt
    │   ├── api_config_CI_Coverage_cannot_test.txt
    │   ├── api_config_CI_Coverage.txt
    │   ├── api_config_CI_Distribute_stable.txt
    │   ├── api_config_CI_py3.txt
    │   ├── api_config_merged_10.txt
    │   ├── api_config_merged_11.txt
    │   ├── api_config_merged_12.txt
    │   ├── api_config_merged_1.txt
    │   ├── api_config_merged_2.txt
    │   ├── api_config_merged_3.txt
    │   ├── api_config_merged_4.txt
    │   ├── api_config_merged_5.txt
    │   ├── api_config_merged_6.txt
    │   ├── api_config_merged_7.txt
    │   ├── api_config_merged_8.txt
    │   ├── api_config_merged_9.txt
    │   ├── api_config_merged_amp.txt
    │   ├── api_config_merged_getset_item_1.txt
    │   ├── api_config_merged_getset_item_2.txt
    │   ├── api_config_merged_getset_item_3.txt
    │   ├── api_config_merged_getset_item_4.txt
    │   ├── api_config_merged_getset_item_5.txt
    │   ├── api_config_merged_getset_item_6.txt
    │   ├── api_config_merged_not_support_sparse.txt
    │   ├── api_config_merged_not_support.txt
    │   ├── api_config_not_support_to_torch_1.txt
    │   ├── api_config_support_to_torch_1.txt
    │   ├── api.yaml
    │   ├── bigtensor_accuracy.txt
    │   ├── bigtensor_paddleonly.txt
    │   ├── cinn_20250217
    │   ├── config_analyzer.py
    │   ├── __init__.py
    │   ├── normalize_origin_api_config.py
    │   ├── __pycache__
    │   ├── test_0_size_20250120
    │   ├── test_20250120
    │   ├── test_big_tensor_20250213
    │   ├── test_log
    │   ├── test_log_0size_accuracy
    │   ├── test_log_0size_paddleonly
    │   ├── test_log_accuracy
    │   ├── test_log_accuracy_20250227
    │   ├── test_log_bigtensor_accuracy
    │   ├── test_log_bigtensor_paddleonly
    │   ├── test_tool.py
    │   └── to_0_size_config.py
    ├── base.py
    ├── __init__.py
    ├── paddle_cinn_vs_dygraph.py
    ├── paddle_only.py
    ├── paddle_to_torch
    │   ├── converter.py
    │   ├── deprecated
    │   ├── __init__.py
    │   ├── mapping.json
    │   ├── __pycache__
    │   └── rules.py
    └── __pycache__
```

## 3. 使用介绍

### 环境配置
运行环境可大致分为**cpu**环境与**gpu**环境，cpu和gpu上运行的结果**可能存在差异**，即存在cpu上能够正确运行，但gpu上报错的情况。因此需要根据需求正确安装环境。

下载链接可参考：https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html

若需要本地编译paddle，可参考链接：https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html

测试CPU除了通过上述链接安装CPU的最新develop包之外，还可使用如下指令设置Paddle工作在CPU模式：
```
paddle.device.set_device("cpu")
```

### 使用说明

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

运行配置时，可能会存在bug导致**测试中断**。PaddleAPITest开发了checkpoint机制，记录“已测试”配置。可以通过如下命令启动批量测试：
```
for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --accuracy=True >> tester/api_config/test_log/log.log 2>&1; done

for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --paddle_only=True >> tester/api_config/test_log/log.log 2>&1; done

for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --paddle_cinn=True >> tester/api_config/test_log/log.log 2>&1; done
```

## 4. 工作进展

1. **配置采集工作**： 已完成，采集到732个API的339.5万个配置，有226个API未采集到配置。没有采集到配置的API暂不做测试。
   
2. **前置引擎支持**： 已初步完成对所有API的引擎支持工作，目前正在开展cpu与gpu环境的全量回测，sparse相关API与内核修复相关，暂时跳过。
   
3. **内核转换机制**： 已初步完成Paddle2Torch内核转换机制的代码编写，后续将开展这一部分的API修复工作，具体细节详见：
   
>https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/dA5upd0dPu/ODBEcpC8QXcAAE
>
>https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/dA5upd0dPu/-75canpiFaJClt
   
5. **0-size向量**：  有少数API已完成修复，大部分API尚未开展
   
6. **特大tensor**：  尚未开展
