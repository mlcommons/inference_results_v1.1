# Qualcomm Cloud AI - MLPerf Inference - SSD-ResNet34

<a name="submit_aedk_20w_singlestream"></a>
## Single Stream

<a name="submit_aedk_20w_singlestream_accuracy"></a>
### Accuracy

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=ssd_resnet34 --scenario=singlestream --mode=accuracy
...
mAP=
</pre>

<a name="submit_aedk_20w_singlestream_performance"></a>
### Performance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=ssd_resnet34 --scenario=singlestream --mode=performance --target_latency=30
</pre>

<a name="submit_aedk_20w_singlestream_power"></a>
### Power

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=ssd_resnet34 --scenario=singlestream --mode=performance --target_latency=30 \
--power=yes --power_server_ip=192.168.0.3 --power_server_port=4949 --sleep_before_ck_benchmark_sec=60
</pre>

<a name="submit_aedk_20w_singlestream_compliance"></a>
### Compliance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=ssd_resnet34 --scenario=singlestream --target_latency=30 \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01
</pre>

## Info

Please contact anton@krai.ai if you have any problems or questions.
