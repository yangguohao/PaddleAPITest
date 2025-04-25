# Paddle2Torch

## 目录

- [模块介绍](#模块介绍)
- [开发文档](#开发文档)
- [贡献指南](#贡献指南)
- [结语](#结语)

## 模块介绍

Paddle2Torch 是一个专注于将 PaddlePaddle API 转换为 PyTorch 对应实现的知识工具库，属于 [PaddleAPITest](https://github.com/PFCCLab/PaddleAPITest) 项目的核心组成模块。本模块通过解析 PaddlePaddle API 调用，使用预定义的转换规则与动态代码生成，实现从 PaddlePaddle 到 PyTorch 的自动转换。转换过程将确保代码的语义一致性。

本模块具有精简强悍的架构，仅由三个组件构成：
- *转换引擎 converter.py*
- *转换规则 rules.py*
- *转换配置 mapping.json*

代码已完全进行解耦，可以非常容易地迁移至其他代码中。本模块通过 **转换配置** 与 **转换规则** 管理 API 映射关系，因此支持开发者灵活扩展新的 API 转换能力。

本模块的典型应用场景包括：模型迁移、跨框架验证、混合编程等，可为深度学习开发者提供跨框架的互操作性解决方案。

## 开发文档

百度内部同学请参考：
- [Paddle2Torch 内核机制开发文档](https://ku.baidu-int.com/d/ODBEcpC8QXcAAE)
- [PaddleAPITest Paddle2Torch 使用文档](https://ku.baidu-int.com/d/-75canpiFaJClt)

## 贡献指南

如果您在使用或测试过程中发现尚未支持的 Paddle API 转换，可以参考以下开发流程进行快速开发，完善 Paddle2Torch 的转换能力。以 paddle.crop 为例：

### 检查支持情况

1. 首先在 mapping.json 中搜索 paddle.crop，查看是否已有相关 API 配置。若存在，可以在全局搜索 API 名称，提取其所有测试配置，进行测试；若无任何搜索结果，说明此 Paddle2Torch 尚未支持转换此 API，需要我们补齐转换能力。此时未搜索到 paddle.crop，开始进行补齐工作。 

### 查询开发资料

2. 在 [paddle 官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html) 中搜索 paddle.crop，对照 API 文档，做好转换能力开发的准备。paddle.crop 的 API 介绍为：

    > paddle.crop(x, shape=None, offsets=None, name=None)
    > 
    > 根据偏移量（offsets）和形状（shape），裁剪输入（x）Tensor。

    飞桨官方开发了 Torch 转 Paddle 的强大代码工具 [PaConvert](https://github.com/PaddlePaddle/PaConvert) ，并且飞桨文档中也有完备的 [PyTorch 最新 release 与 Paddle develop API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html) ，详细说明了哪些 API 可以互相转换，不能转换的原因与可能的解决办法是什么。我们可以先查询并参考这些资料👆。

3. 在 [PyTorch 最新 release 与 Paddle develop API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html) 中搜索 paddle.crop，查看是否有符合条目。若存在，则分别点击 **Torch API** 和 **详细对比**，仔细阅读内容，思考其提供的方案是否可行；若没有发现任何条目，说明此 API 是比较少见的类型、或是新 API，需要我们再次仔细阅读 API 文档描述，思考并查询对应的 Torch API 可能是什么。paddle.crop 没有现成的转换方案，需要进一步寻找。


4. 在 [PyTorch 官网](https://pytorch.org/docs/stable/index.html) 中搜索 crop，仅找到图像操作的 API： [torchvision.transforms.functional.crop](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html) ，不太符合我们想要的 Torch API。

   经查阅资料，能够实现 paddle.crop 表现的有 torch.narrow 或直接使用切片操作（Torch 重载了 [] 操作符）。前者仅能实现单维度裁剪，实现多维度需要进行循环，较为复杂；后者则类似于 numpy 风格的切片，虽然也需要循环，但可以压缩至一行，非常 pythonic。因此决定使用 Torch 的切片操作模拟 paddle.crop 的表现。

### 组织编写思路

5. 由于构造切片所用的 slices 参数需要使用循环，且属于特殊操作（不属于调用 Torch API），因此需要继承 BaseRule，编写新的 Rule 类。如果能够通过 **直接参数映射** 或 **组合映射** 方式实现的话，建议最好在 mapping.json 中编写配置即可，可直接跳转至 [编写转换配置](#编写转换配置) 章节。

6. paddle.crop 的参数介绍中详细介绍了不同参数的类型、默认值等，我们需要支持所有的配置情况，并考虑到参数缺省。参数介绍如下：

    > **x** (Tensor) - 1-D 到 6-D Tensor，数据类型为 float32、float64、int32 或者 int64。
    > 
    > **shape** (list|tuple|Tensor，可选) - 输出 Tensor 的形状，数据类型为 int32。如果是列表或元组，则其长度必须与 x 的维度大小相同，如果是 Tensor，则其应该是 1-D Tensor。当它是列表时，每一个元素可以是整数或者形状为[]的 0-D Tensor。含有 Tensor 的方式适用于每次迭代时需要改变输出形状的情况。
    > 
    > **offsets** (list|tuple|Tensor，可选) - 每个维度上裁剪的偏移量，数据类型为 int32。如果是列表或元组，则其长度必须与 x 的维度大小相同，如果是 Tensor，则其应是 1-D Tensor。当它是列表时，每一个元素可以是整数或者形状为[]的 0-D Tensor。含有 Tensor 的方式适用于每次迭代的偏移量（offset）都可能改变的情况。默认值：None，每个维度的偏移量为 0。

    可以看到，paddle.crop 的 shape、offsets 参数具有非常丰富的形式，可以是 *缺省*、*列表或元组*、*1-D Tensor*，列表或元组可以由 *int* 或 *0-D Tensor* 组成。

7. 在测试配置中搜索 paddle.crop ，可以看到 shape 中允许 -1，说明该维度的大小由 x 和 offsets 推断，我们也需要支持此种配置。

    ```python
    paddle.crop(x=Tensor([2, 3, 3, 3],"float64"), shape=list[2,1,-1,2,], offsets=list[0,0,1,1,], )
    ```

### 编写转换代码

8. 在编写代码前，测试环境已经将 paddle.crop 的所有参数放置于变量 `arg` 、`kwargs` 和 执行环境 `locals()` 中，我们可以通过 `kwargs.get('var')` 、 `locals().get('var')` 或直接使用 `var` 获取参数（ 未提供 `var` 参数时直接访问会抛出 `NameError` 错误，而 `get()` 获取可以设定默认值）。

    首先单独构造出 slices 可用的 shape 与 offsets 参数，使用 list 来表示（默认所有参数均是符合文档描述的，不需要再验证和抛出错误）：

    ```python
    ndim = x.dim()
    offsets = locals().get('offsets')
    shape = locals().get('shape')

    if offsets is None:
        offsets = [0] * ndim
    elif isinstance(offsets, (list, tuple)):
        offsets = [o.item() if isinstance(o, torch.Tensor) else int(o) for o in offsets]
    elif isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()

    if shape is None:
        shape = [x.size(i) - offsets[i] for i in range(ndim)]
    elif isinstance(shape, (list, tuple)):
        shape = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in shape]
    elif isinstance(shape, torch.Tensor):
        shape = shape.tolist()
    ```

    推断并替换 shape 中所有 -1 值。

    ```python
    shape = [x.size(i) - offsets[i] if s == -1 else s for i, s in enumerate(shape)]
    ```

    根据 shape 与 offsets 构造 slices 参数：

    ```python
    slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(ndim)]
    ```

    使用 Torch 切片操作，将结果保存至 result 中（ x 一定存在于 `locals()` 中，不需要再使用 `get()` ）：

    ```python
    result = x[slices]
    ```

    至此，转换代码编写完成.

### 测试转换代码

9. 为了验证转换代码的正确性，我们可以编写一些简单的测试用例去测试它，避免到了测试执行时才报错：

    ```python
    import torch


    def torch_crop(x, shape=None, offsets=None):
        ndim = x.dim()
        if offsets is None:
            offsets = [0] * ndim
        elif isinstance(offsets, (list, tuple)):
            offsets = [o.item() if isinstance(o, torch.Tensor) else int(o) for o in offsets]
        elif isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()

        if shape is None:
            shape = [x.size(i) - offsets[i] for i in range(ndim)]
        elif isinstance(shape, (list, tuple)):
            shape = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in shape]
        elif isinstance(shape, torch.Tensor):
            shape = shape.tolist()

        shape = [x.size(i) - offsets[i] if s == -1 else s for i, s in enumerate(shape)]
        slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(ndim)]

        return x[slices]


    x = torch.arange(16).reshape(4, 4)
    print(torch_crop(x, [2, 2], [1, 1]))

    x = torch.arange(27).reshape(3, 3, 3)
    print(torch_crop(x, [-1, 2, 2], [0, 1, 0]))

    x = torch.arange(16).reshape(4, 4)
    print(torch_crop(x, torch.tensor([2, 2]), torch.tensor([1, 1])))

    x = torch.arange(16).reshape(4, 4)
    print(torch_crop(x, [torch.tensor(2), 2], [torch.tensor(1), 1]))

    x = torch.arange(16).reshape(4, 4)
    print(torch_crop(x))

    ```

    测试结果符合预期，我们成功地使用了 Torch 模拟出 Paddle API 的所有表现了！现在可以开始编写 Rule 类了！

### 编写转换配置

10. 若仅需要编写转换配置，需在 mapping.json 的相应条目（去掉 paddle. 后的字典序）下编写，编写规则为：

    ```json
        "<api_name>": {
            "torch_api": "torch api 名称（torch_api 与 composite_steps 必须定义其一）",
            "paddle_torch_args_map": {
                "_description": "参数名映射字典，键对应 paddle，值对应 torch",
            },
            "torch_args": [
                "torch api 位置参数列表, 变量名可使用 {} 环绕，字符串的引号请使用 \\ 转义，也可以直接设为常值"
            ],
            "torch_kwargs": {
                "_description": "torch api 关键字参数字典，与 torch_args 类似"
            },
            "composite_steps": [
                "当需要多个 torch api 组合实现时，定义步骤列表，每行的执行结果将被赋值给 _tmp_i，可通过 {i} 访问",
                {
                    "torch_api": "torch api",
                    "torch_args": [
                        "torch api 位置参数列表，可以使用 {i} 代表中间变量"
                    ],
                    "torch_kwargs": {
                        "_description": "torch api 关键字参数字典，与 torch_args 类似"
                    }
                }
            ]
        }
    ```

11. 若需要编写转换代码，既需要在 mapping.json 中注册，也需要在 rules.py 中定义类。注册规则为：
    
    ```json
        "<api_name>": {
            "Rule": "自定义的 Rule 类的类名"
        }
    ```

    此外，也可以添加更多的常规配置，以减少 Rule 类代码的编写量（需要主动使用 apply_generic() 方法获取 code ）：

    ```json
        "<api_name>": {
            "Rule": "自定义的 Rule 类的类名",
            "torch_api": "torch api 名称",
            "paddle_torch_args_map": {
                "_description": "参数名映射字典，键对应 paddle，值对应 torch"
            },
            "set_default": {
                "_description": "默认值设置字典，键为参数名，值为默认值"
            },
            "import": [
                "需要导入的模块名列表"
            ]
        }
    ```

    对于 paddle.crop 而言，直接在 mapping.json 的 "c" 条目下注册 Rule 类：

    ```json
        "paddle.crop":{
            "Rule": "CropRule"
        },
    ```
12. Rule 类的定义需要继承自抽象基类 BaseRule，并实现 apply() 方法，否则无法执行转换。基类定义为：

    ```python
    class BaseRule(ABC):
    """转换规则的抽象基类"""

    @abstractmethod
    def apply(self, paddle_api: str) -> ConvertResult:
        pass
    ```

    在 rules.py 的 #c 注释下编写 Rule 类 CropRule：

    ```python
    class CropRule(BaseRule):
        def apply(self, paddle_api: str) -> ConvertResult:
            impl = """
    ndim = x.dim()
    offsets = locals().get('offsets')
    shape = locals().get('shape')

    if offsets is None:
        offsets = [0] * ndim
    elif isinstance(offsets, (list, tuple)):
        offsets = [o.item() if isinstance(o, torch.Tensor) else int(o) for o in offsets]
    elif isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()

    if shape is None:
        shape = [x.size(i) - offsets[i] for i in range(ndim)]
    elif isinstance(shape, (list, tuple)):
        shape = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in shape]
    elif isinstance(shape, torch.Tensor):
        shape = shape.tolist()

    shape = [x.size(i) - offsets[i] if s == -1 else s for i, s in enumerate(shape)]
    slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(ndim)]

    result = x[slices]
    """
            code = impl.splitlines()
            return ConvertResult.success(paddle_api, code, "result")
    ```

### 运行测试配置

13. 全局搜索 paddle.crop ，将所有相关测试配置移至临时文件中，然后运行 accuracy 测试命令：

    ```shell
    python engine.py --accuracy=True --api_config_file="tester/api_config/api_config_merged_temp.txt"
    ```

    最终测试配置全部通过，结果位于 test_log\api_config_pass.txt，合并至通过 accuracy 测试的 api_config_accuracy_*.txt 中。

### 其他情况

14.  如果 Paddle API 的行为实在难以通过 Torch 表达，可暂时不对其进行支持。可为其注册 ErrorRule 类或直接不做处理，并将所有相关配置合并至未通过 accuracy 测试的 api_config_paddleonly_*.txt 中。

## 结语

感谢同学们仔细阅读 README 至此，如果您有任何修改建议，或问题修复、转换补齐的想法，请提交 Issue 与 PR ，并 at @cangtianhuang 进行 Review

也可以直接发送至开发者邮箱: 1903374751@qq.com / l1903374751@gmail.com

非常感谢以下贡献人员:

@wanghuancoder @cangtianhuang @mzj104 @Cutelemon6 @cszdrg @yuwu46
