# SSD-MobileNet Docker image

## Benchmark

### Offline

#### Accuracy

##### `r282_z93_q1`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-mobilenet.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.6 --model=ssd_mobilenet \
--mode=accuracy --scenario=offline --target_qps=19500"
```

##### `r282_z93_q5`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-mobilenet.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.6 --model=ssd_mobilenet \
--mode=accuracy --scenario=offline --target_qps=97500"
```

#### Performance

##### `r282_z93_q1`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-mobilenet.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.6 --model=ssd_mobilenet \
--mode=performance --scenario=offline --target_qps=19500"
```

##### `r282_z93_q5` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-mobilenet.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.6 --model=ssd_mobilenet \
--mode=performance --scenario=offline --target_qps=97500"
```

#### Power

##### `r282_z93_q1` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-mobilenet.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.6 --model=ssd_mobilenet \
--mode=performance --scenario=offline --target_qps=19500 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

##### `r282_z93_q5`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-mobilenet.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.6 --model=ssd_mobilenet \
--mode=performance --scenario=offline --target_qps=97500 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```
