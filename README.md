# PaddleAPITest
******
## 1. 项目背景
公司内部业务，或者Paddle Issue，偶尔会反馈Paddle API正确性的问题，经过梳理大致有3类问题：
>1）API精度不正确；
>2）一些特大Tensor，尤其是numel超过int32上限的Tensor计算异常；
>3）0-Size（numel为0的Tensor） Tensor不支持。
API正确性是Paddle质量的基石，影响业务训练、推理，影响用户对Paddle的信赖。至关重要。我们需要统一排查问题API、问题case，统一修复。

为此，我们开始着手发了PaddleAPITest用于排查问题API、问题case。主要工作思路如下：
1. 在Paddle开发Trace API机制，用于抓取API调用配置，下面是一个例子：
    paddle.concat(tuple(Tensor([31376, 768],"float32"),Tensor([1, 768],"float32"),), axis=0, )
2. 在所有Paddle单元测试（CI）、集成测试（CE）流水线中，抓取所有Paddle API的调用配置，形成了PaddleAPITest/tester/api_config下以“api_config_CI”，“api_config_CE”开头的配置集。对以上配置集进行去重、排序、梳理得到了以“api_config_merged”开头的配置集。
3. 在 PaddleAPITest 中开发一套“引擎”，加载配置集，初始化相应Tensor，调用相应API执行前/反向测试。
4. 在 PaddleAPITest 中开发一套“转换工具”，在调用Paddle API测试的同时，等同的调用Torch API，做精度对比测试。
5. 对采集到的配置集进行shape篡改，得到了“bigtensor”、“0sizetensor”开头的配置集。
6. 通过与Torch对比，如果出现以下情况则认为Paddle API有必要确认是否正确并修复：
>a. 精度diff
>b. Torch正常，Paddle报错
>c. Torch正常，Paddle CoreDump或CUDA Error

## 2. 使用介绍

### 环境配置
运行环境可大致分为cpu环境与gpu环境，cpu和gpu上运行的结果可能存在差异，即存在cpu上能够正确运行，但gpu上报错的情况。因此需要根据需求正确安装环境。下载链接可参考：https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html

若需要本地编译paddle，可参考链接：https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html

### 使用说明

所有测试前，**必须创建**一个目录：PaddleAPITest/tester/api_config/test_log/，用于存放测试所产生的测试结果和checkpoint。

PaddleAPITest目前支持paddle_only、accuracy、paddle_cinn三种测试：
>paddle_only，用于单纯把配置在Paddle动态图跑一遍，验证PaddleAPITest 引擎是否支持该配置。
>accuracy，用于将Paddle API的前反向与Torch的前反向做精度对比测试。
>paddle_cinn，用于Paddle动态图与Paddle静态图编译器做精度对比测试。

当测试单个配置时，可使用下面的代码，--api_config中输入待测试的配置内容：
仅测试paddle是否支持：
    python engine.py --paddle_only=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'
测试输出是否准确：
    python engine.py --accuracy=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'
动态图和静态图测试：
    python engine.py --paddle_cinn=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'

值得注意的是配置txt中统一使用双引号"，因此建议--api_config=''使用单引号，或在配置中手动添加转义斜杠\




for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --accuracy=True >> tester/api_config/test_log/log.log 2>&1; done

for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --paddle_only=True >> tester/api_config/test_log/log.log 2>&1; done

for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --paddle_cinn=True >> tester/api_config/test_log/log.log 2>&1; done


python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --accuracy=True > tester/api_config/test_log/log.log 2>&1

python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_only=True > tester/api_config/test_log/log.log 2>&1

python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_cinn=True > tester/api_config/test_log/log.log 2>&1

