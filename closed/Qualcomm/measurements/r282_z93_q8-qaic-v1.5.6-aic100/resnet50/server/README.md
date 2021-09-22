# Qualcomm Cloud AI - MLPerf Inference - Image Classification

<a name="submit_r282_z93_q5_server"></a>
## Server

<a name="submit_r282_z93_q5_server_accuracy"></a>
### Accuracy

<pre>
<b>[anton@ax530b-03-giga ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.00 --model=resnet50 --scenario=server \
--mode=accuracy --target_qps=78500 --dataset_size=50000 --buffer_size=5000
</pre>

<a name="submit_r282_z93_q5_server_performance"></a>
### Performance

<pre>
<b>[anton@ax530b-03-giga ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.00 --model=resnet50 --scenario=server \
--mode=performance --target_qps=78500 --dataset_size=50000 --buffer_size=1024
</pre>

<a name="submit_r282_z93_q5_server_power"></a>
### Power

<pre>
<b>[anton@ax530b-03-giga ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.00 --model=resnet50 --scenario=server \
--mode=performance --target_qps=78500 --dataset_size=50000 --buffer_size=1024 \
--power=yes --power_server_ip=192.168.0.3 --power_server_port=4949 --sleep_before_ck_benchmark_sec=90
</pre>

<a name="submit_r282_z93_q5_server_compliance"></a>
### Compliance

<pre>
<b>[anton@ax530b-03-giga ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.00 --model=resnet50 --scenario=server \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01 --target_qps=78500 --dataset_size=50000 --buffer_size=1024
</pre>

## Info

Please contact anton@krai.ai if you have any problems or questions.
