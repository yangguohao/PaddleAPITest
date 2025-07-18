export CUDA_VISIBLE_DEVICES=1
# sleep 3h
for i in {1..10000}; do python engine.py --api_config_file=test_pipline/V1/gpu_prof/gpu_bigtensor_accuracy_errorconfig_2.txt --paddle_torch_gpu_performance=True >> test_pipline/V1/gpu_prof/gpu_bigtensor_accuracy_errorconfig_2.log 2>&1; done
