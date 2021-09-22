# Qualcomm Cloud AI - MLPerf ResNet50 Docker image

## Benchmark

### Offline

#### Accuracy

##### `r282_z93_q1`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.9 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=resnet50 \
--mode=accuracy --scenario=singlestream --target_latency=0.5"
```

##### `r282_z93_q5`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.9 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=resnet50 \
--mode=accuracy --scenario=singlestream --target_latency=0.5"
```

#### Performance

##### `r282_z93_q1`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.9 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=resnet50 \
--mode=performance --scenario=singlestream --target_latency=0.5"
```

##### `r282_z93_q5` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.9 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=resnet50 \
--mode=performance --scenario=singlestream --target_latency=0.5"
```

#### Power

##### `r282_z93_q1` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.9 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=resnet50 \
--mode=performance --scenario=singlestream --target_latency=0.5 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

##### `r282_z93_q5`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.9 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=resnet50 \
--mode=performance --scenario=singlestream --target_latency=0.5 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```
