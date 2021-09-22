# Qualcomm Cloud AI - MLPerf Inference - Image Classification

<a name="submit_aedk_20w_offline"></a>
## Offline

<a name="submit_aedk_20w_offline_accuracy"></a>
### Accuracy

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=resnet50 --mode=accuracy --scenario=offline
...
accuracy=76.002%, good=38001, total=50000
</pre>

<a name="submit_aedk_20w_offline_performance"></a>
### Performance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=resnet50 --mode=performance --scenario=offline --target_qps=9666 
</pre>

<a name="submit_aedk_20w_offline_power"></a>
### Power

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=resnet50 --mode=performance --scenario=offline --target_qps=9666 \
--power=yes --power_server_ip=192.168.0.3 --power_server_port=4949 --sleep_before_ck_benchmark_sec=60
</pre>

<a name="submit_aedk_20w_offline_compliance"></a>
### Compliance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk_20w --sdk=1.5.00 --model=resnet50 --scenario=offline --target_qps=9666 \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01
</pre>

## Info

Please contact anton@krai.ai if you have any problems or questions.
