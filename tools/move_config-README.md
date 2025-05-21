在完善paddle2torch转换过程中，需要大量的搜索api，复制配置、移动配置以及删除配置，手动进行这些过程存在繁琐、易出错的问题。为了解决上述问题，标准化移动配置的流程，编写了自动移动配置工具。

-----------
**使用介绍**

代码可以直接在/PaddleAPITest/根目录下运行


config参数填入api名称，例如：paddle.numel

dst参数填入输出到的txt名称，若不指定会默认输出到mytmp.txt

暂时不想移除原有的配置：
`python tools/move_config.py --config='your api' --dst='tmp.txt'`

移除原有的配置：
`python tools/move_config.py --config='your api' --dst='tmp.txt' --remove`