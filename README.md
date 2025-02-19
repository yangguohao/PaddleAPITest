# PaddleAPITest

for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --accuracy=True >> tester/api_config/test_log/log.log 2>&1; done

for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --paddle_only=True >> tester/api_config/test_log/log.log 2>&1; done

for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_merge.txt --paddle_cinn=True >> tester/api_config/test_log/log.log 2>&1; done


python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --accuracy=True > tester/api_config/test_log/log2.log 2>&1

python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_only=True > tester/api_config/test_log/log2.log 2>&1

python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_cinn=True > tester/api_config/test_log/log2.log 2>&1


python engine.py --accuracy=True --api_config="paddle.abs(Tensor([1, 100],\"float64\"), )"

python engine.py --paddle_only=True --api_config="paddle.abs(Tensor([1, 100],\"float64\"), )"

python engine.py --paddle_cinn=True --api_config="paddle.abs(Tensor([1, 100],\"float64\"), )"

