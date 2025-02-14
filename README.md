# PaddleAPITest
python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_cinn=True > tester/api_config/test_log/log2.log 2>&1
python engine.py --paddle_only=True --api_config="paddle.Tensor.__setitem__(Tensor([1, 1, 64, 64],\"float32\"), tuple(0,0,slice(None,64,None),slice(None,64,None),), Tensor([64, 64],\"float32\"), )"

for i in {1..10000}; do python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_cinn.txt --paddle_cinn=True >> tester/api_config/test_log/log.log 2>&1; done