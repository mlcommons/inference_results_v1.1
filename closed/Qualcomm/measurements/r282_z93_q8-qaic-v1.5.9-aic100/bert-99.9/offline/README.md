# Qualcomm Cloud AI - MLPerf BERT Docker image

## Benchmark

### Offline

#### Accuracy

##### `r282_z93_q1`

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=accuracy --scenario=offline --override_batch_size=4096 --target_qps=670"
```

###### precision fp16

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=accuracy --scenario=offline --override_batch_size=4096 --target_qps=300"
```

##### `r282_z93_q5`


###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=accuracy --scenario=offline --override_batch_size=4096 --target_qps=3350"
```

###### precision fp16

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=accuracy --scenario=offline --override_batch_size=4096 --target_qps=1500"
```

##### `r282_z93_q8`

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=accuracy --scenario=offline --override_batch_size=4096 --target_qps=5360"
```

###### precision fp16

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=accuracy --scenario=offline --override_batch_size=4096 --target_qps=2400"
```

##### `g292_z43_q16`

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=accuracy --scenario=offline --override_batch_size=4096 --target_qps=10600"
```


###### precision fp16

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=accuracy --scenario=offline --override_batch_size=4096 --target_qps=5000"
```

#### Performance

##### `r282_z93_q1`

###### precision mixed
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=670"
```

###### precision fp16
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=300"
```

##### `r282_z93_q5` [optional]

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=3350"
```

###### precision fp16
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=1500"
```

##### `r282_z93_q8` [optional]


###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=5360"
```

###### precision fp16
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=2400"
```

##### `g292_z43_q16` [optional]

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=10600"
```

###### precision fp16
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=5000"
```

#### Power

##### `r282_z93_q1` [optional]

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=670 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

###### precision fp16
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q1 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=300 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

##### `r282_z93_q5`

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=3350 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

###### precision fp16
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q5 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=1500 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4956"
```

##### `r282_z93_q8`

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=4800 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4959"
```

###### precision fp16
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=r282_z93_q8 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=2400 \
--power=yes --power_server_ip=10.222.154.58 --power_server_port=4959"
```

##### `g292_z43_q16`

###### precision mixed

```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.9 --model=bert --model_extra_tags=precision.mixed \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=10600 \
--power=yes --power_server_ip=10.222.147.109 --power_server_port=4953"
```

###### precision fp16
```
docker run --privileged \
--user=krai:kraig --group-add $(cut -d: -f3 < <(getent group qaic)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment \
--rm krai/mlperf.bert.centos7:1.5.9 \
"ck run cmdgen:benchmark.packed-bert.qaic-loadgen --verbose \
--sut=g292_z43_q16 --sdk=1.5.9 --model=bert --model_extra_tags=precision.fp16 \
--mode=performance --scenario=offline --override_batch_size=4096 --target_qps=5000 \
--power=yes --power_server_ip=10.222.147.109 --power_server_port=4953"
```
