# Qualcomm Cloud AI - MLPerf SSD-ResNet34 Docker image

## Benchmark

### Singlestream

#### Accuracy

##### `r282_z93_q1`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.9 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=ssd_resnet34 \
--mode=accuracy --scenario=singlestream --target_latency=12.5"
```

##### `r282_z93_q5`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.9 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=ssd_resnet34 \
--mode=accuracy --scenario=singlestream --target_latency=12.5"
```


#### Performance

##### `r282_z93_q1`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.9 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=ssd_resnet34 \
--mode=performance --scenario=singlestream --target_latency=12.5"
```

##### `r282_z93_q5` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.9 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=ssd_resnet34 \
--mode=performance --scenario=singlestream --target_latency=12.5"
```


#### Power

##### `r282_z93_q1` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.9 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=ssd_resnet34 \
--mode=performance --scenario=singlestream --target_latency=12.5 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

##### `r282_z93_q5`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.9 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=ssd_resnet34 \
--mode=performance --scenario=singlestream --target_latency=12.5 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```
