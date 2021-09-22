# Qualcomm Cloud AI - MLPerf Inference - SSD-ResNet34

<a name="submit_aedk_20w_offline"></a>
## Offline

<a name="submit_aedk_20w_offline_accuracy"></a>
### Accuracy

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=ssd_resnet34 --scenario=offline --mode=accuracy
...
mAP=...
</pre>

<a name="submit_aedk_20w_offline_performance"></a>
### Performance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=ssd_resnet34 --scenario=offline --mode=performance --target_qps=199
</pre>

<a name="submit_aedk_20w_offline_power"></a>
### Power

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=ssd_resnet34 --scenario=offline --mode=performance --target_qps=199 \
--power=yes --power_server_ip=192.168.0.3 --power_server_port=4949 --sleep_before_ck_benchmark_sec=60
</pre>

<a name="submit_aedk_20w_offline_compliance"></a>
### Compliance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=ssd_resnet34 --scenario=offline --target_qps=199 \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01
</pre>

## Info

Please contact anton@krai.ai if you have any problems or questions.
