# Qualcomm Cloud AI - MLPerf Inference - Image Classification

<a name="submit_aedk_16nsp_singlestream"></a>
## Single Stream

<a name="submit_aedk_16nsp_singlestream_accuracy"></a>
### Accuracy

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=resnet50 --mode=accuracy --scenario=singlestream
...
accuracy=75.942%, good=37971, total=50000
</pre>

<a name="submit_aedk_16nsp_singlestream_performance"></a>
### Performance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=resnet50 --mode=performance --scenario=singlestream --target_latency=1
</pre>

<a name="submit_aedk_16nsp_singlestream_power"></a>
### Power

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=resnet50 --mode=performance --scenario=singlestream --target_latency=1 \
--power=yes --power_server_ip=192.168.0.3 --power_server_port=4949 --sleep_before_ck_benchmark_sec=30
</pre>

<a name="submit_aedk_16nsp_singlestream_compliance"></a>
### Compliance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=resnet50 --scenario=singlestream --target_latency=1 \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01
</pre>

## Info

Please contact anton@krai.ai if you have any problems or questions.
