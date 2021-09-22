# Qualcomm Cloud AI - MLPerf SSD-ResNet34 Docker image

## Benchmark

### Server

#### Accuracy

##### `r282_z93_q8`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=ssd_resnet34 \
--mode=accuracy --scenario=server --target_qps=3380"
```

##### `g292_z43_q16`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.6 --model=ssd_resnet34 \
--mode=accuracy --scenario=server --target_qps=6866"
```

#### Performance

##### `r282_z93_q8` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=ssd_resnet34 \
--mode=performance --scenario=server --target_qps=3380"
```

##### `g292_z43_q16` [optional]

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.6 --model=ssd_resnet34 \
--mode=performance --scenario=server --target_qps=6866"
```

#### Power

##### `r282_z93_q8`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.6 --model=ssd_resnet34 \
--mode=performance --scenario=server --target_qps=3380 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4959"
```

##### `g292_z43_q16`

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.ssd-resnet34.centos7:1.5.6 \
"ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.6 --model=ssd_resnet34 \
--mode=performance --scenario=server --target_qps=6866 \
--power=yes --power_server_ip=10.222.147.109 --power_server_port=4953"
```
