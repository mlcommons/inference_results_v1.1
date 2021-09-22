# Qualcomm Cloud AI - MLPerf ResNet50 Docker image

## Benchmark

### Server

#### Accuracy

##### `r282_z93_q8`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=resnet50 \
--mode=accuracy --scenario=server --target_qps=156666"
```

##### `g292_z43_q16`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.6 --model=resnet50 \
--mode=accuracy --scenario=server --target_qps=313333"
```

#### Performance

##### `r282_z93_q8`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=server --target_qps=16666"
```

##### `g292_z43_q16`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=server --target_qps=33333"
```

#### Power (full)

##### `r282_z93_q8`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.resnet50.full.centos7:1.5.6 \
"ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=resnet50 \
--mode=performance --scenario=server --target_qps=133133 \
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
--mode=performance --scenario=server --target_qps=310000 \
--power=yes --power_server_ip=10.222.147.109 --power_server_port=4953"
```
