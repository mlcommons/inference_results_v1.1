# Qualcomm Cloud AI - MLPerf ResNet50 Docker image

## Benchmark

### Offline

#### Accuracy

##### `r282_z93_q1`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.6 --model=resnet50 \
--mode=accuracy --scenario=offline --target_qps=22222"
```

##### `r282_z93_q5`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.6 --model=resnet50 \
--mode=accuracy --scenario=offline --target_qps=111111"
```

##### `r282_z93_q8`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=resnet50 \
--mode=accuracy --scenario=offline --target_qps=166666"
```

##### `g292_z43_q16`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.6 --model=resnet50 \
--mode=accuracy --scenario=offline --target_qps=333333"
```

#### Performance

##### `r282_z93_q1`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=offline --target_qps=22222"
```

##### `r282_z93_q5` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=offline --target_qps=111111"
```

##### `r282_z93_q8` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=offline --target_qps=166666"
```

##### `g292_z43_q16` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=offline --target_qps=333333"
```

#### Power

##### `r282_z93_q1` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=offline --target_qps=22222 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

##### `r282_z93_q5`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=offline --target_qps=111111 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

##### `r282_z93_q8`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=offline --target_qps=166666 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4959"
```

##### `g292_z43_q16`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=offline --target_qps=333333 \
--power=yes --power_server_ip=10.222.147.109 --power_server_port=4953"
```
