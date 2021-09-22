# Qualcomm Cloud AI - MLPerf Inference - Language

1. [Installation](#installation)
    1. [Install system-wide prerequisites](#install_system)
    1. [Install CK](#install_ck)
    1. [Set platform scripts](#set_platform_scripts)
    1. [Detect Python](#detect_python)
    1. [Detect GCC](#detect_gcc)
    1. [Set up CMake](#install_cmake)
    1. [Install Python dependencies](#install_python_deps)
    1. [Install the MLPerf Inference repo](#install_inference_repo)
    1. [Prepare the SQuAD validation dataset](#prepare_squad)
    1. [Prepare the BERT Large model](#prepare_bert_large)
1. [Benchmark](#benchmark)
    1. [Accuracy](#benchmark_accuracy)
    1. [Performance](#benchmark_performance)

<a name="installation"></a>
# Installation

Tested on a ([Gigabyte R282-Z93](https://www.gigabyte.com/Enterprise/Rack-Server/R282-Z93-rev-100)) server with CentOS 7.9 and QAIC Platform SDK 1.5.9:

<pre><b>[anton@dyson ~]&dollar;</b> rpm -q centos-release
centos-release-7-9.2009.1.el7.centos.x86_64</pre>

<pre><b>[anton@dyson ~]&dollar;</b> uname -a
Linux dyson.localdomain 5.4.1-1.el7.elrepo.x86_64 #1 SMP Fri Nov 29 10:21:13 EST 2019 x86_64 x86_64 x86_64 GNU/Linux</pre>

<pre><b>[anton@dyson ~]&dollar;</b> cat /opt/qti-aic/versions/platform.xml</pre>
```
<versions>
        <ci_build>
           <base_name>AIC</base_name>
           <base_version>1.5</base_version>
           <build_id>9</build_id>
        </ci_build>
        </versions>
```

<a name="install_system"></a>
## Install system-wide prerequisites

**NB:** Run the below commands with `sudo` or as superuser.

<a name="install_system_centos7"></a>
### CentOS 7

#### Generic

<pre>
<b>[anton@dyson ~]&dollar;</b> sudo yum upgrade -y
<b>[anton@dyson ~]&dollar;</b> sudo yum install -y \
make which patch vim git wget zip unzip openssl-devel bzip2-devel libffi-devel
<b>[anton@dyson ~]&dollar;</b> sudo yum clean all
</pre>

#### dnf  ("the new yum"!)

<pre>
<b>[anton@dyson ~]&dollar;</b> sudo yum install -y dnf
</pre>


#### Python 3.6 (default)

<pre>
<b>[anton@dyson ~]&dollar;</b> sudo dnf install -y python3 python3-pip python3-devel
<b>[anton@dyson ~]&dollar;</b> python3 --version
Python 3.6.8
</pre>

#### Python 3.8 (optional; required only for power measurements)

<pre>
<b>[anton@dyson ~]&dollar;</b> sudo su
<b>[root@dyson anton]#</b> export PYTHON_VERSION=3.8.12
<b>[root@dyson anton]#</b> cd /usr/src \
&& wget https://www.python.org/ftp/python/&dollar;{PYTHON_VERSION}/Python-&dollar;{PYTHON_VERSION}.tgz \
&& tar xzf Python-&dollar;{PYTHON_VERSION}.tgz \
&& rm -f Python-&dollar;{PYTHON_VERSION}.tgz \
&& cd /usr/src/Python-&dollar;{PYTHON_VERSION} \
&& ./configure --enable-optimizations && make -j 32 altinstall \
&& rm -rf /usr/src/Python-&dollar;{PYTHON_VERSION}*
<b>[root@dyson ~]#</b> exit
exit
<b>[anton@dyson ~]&dollar;</b> python3.8 --version
Python 3.8.12
</pre>

#### GCC 10

<pre>
<b>[anton@dyson ~]&dollar;</b> sudo yum install -y centos-release-scl
<b>[anton@dyson ~]&dollar;</b> sudo yum install -y scl-utils
<b>[anton@dyson ~]&dollar;</b> sudo yum install -y devtoolset-10
<b>[anton@dyson ~]&dollar;</b> echo "source scl_source enable devtoolset-10" >> ~/.bashrc
<b>[anton@dyson ~]&dollar;</b> source ~/.bashrc
</pre>

##### `gcc`

<pre>
<b>[anton@dyson ~]&dollar;</b> scl enable devtoolset-10 "gcc --version"
gcc (GCC) 10.2.1 20210130 (Red Hat 10.2.1-11)
Copyright (C) 2020 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
</pre>

##### `g++`

<pre>
<b>[anton@dyson ~]&dollar;</b> scl enable devtoolset-9 "g++ --version"
g++ (GCC) 10.2.1 20210130 (Red Hat 10.2.1-11)
Copyright (C) 2020 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
</pre>

<a name="install_ck"></a>
## Install [Collective Knowledge](http://cknowledge.org/) (CK)

<pre>
<b>[anton@dyson ~]&dollar;</b> export CK_PYTHON=`which python3`
<b>[anton@dyson ~]&dollar;</b> &dollar;CK_PYTHON -m pip install --ignore-installed pip setuptools testresources --user --upgrade
<b>[anton@dyson ~]&dollar;</b> &dollar;CK_PYTHON -m pip install ck==1.55.5
<b>[anton@dyson ~]&dollar;</b> echo 'export PATH=&dollar;HOME/.local/bin:&dollar;PATH' >> &dollar;HOME/.bashrc
<b>[anton@dyson ~]&dollar;</b> source &dollar;HOME/.bashrc
<b>[anton@dyson ~]&dollar;</b> ck version
V1.55.5
</pre>

<a name="install_ck_repos"></a>
## Install CK repositories

<pre>
<b>[anton@dyson ~]&dollar;</b> ck pull repo --url=https://github.com/krai/ck-qaic
</pre>


<a name="set_platform_scripts"></a>
## Set platform scripts

### `r282_z93_q5`: use QAIC settings (ECC on)

<pre>
<b>[anton@dyson ~]&dollar;</b> ck detect platform.os --platform_init_uoa=qaic

OS CK UOA:            linux-64 (4258b5fe54828a50)

OS name:              CentOS Linux 7 (Core)
Short OS name:        Linux 5.4.1
Long OS name:         Linux-5.4.1-1.el7.elrepo.x86_64-x86_64-with-centos-7.9.2009-Core
OS bits:              64
OS ABI:               x86_64

Platform init UOA:    qaic

<b>[anton@dyson ~]&dollar;</b> cat $(ck find repo:local)/cfg/local-platform/.cm/meta.json
{
  "platform_init_uoa": {
    "linux-64": "qaic"
  }
}
</pre>


### `aedk`: use AEDK settings

<pre>
<b>[anton@aedk3 ~]&dollar;</b> ck detect platform.os --platform_init_uoa=aedk

OS CK UOA:            linux-64 (4258b5fe54828a50)

OS name:              CentOS Linux 8 (Core)
Short OS name:        Linux 4.19.81
Long OS name:         Linux-4.19.81-aarch64-with-centos-8.0.1905-Core
OS bits:              64
OS ABI:               aarch64

Platform init UOA:    aedk

<b>[anton@aedk3 ~] ~]&dollar;</b> cat $(ck find repo:local)/cfg/local-platform/.cm/meta.json
{
  "platform_init_uoa": {
    "linux-64": "aedk"
  }
}
</pre>

#### Install AEDK specific dependencies
<pre>
curl https://sh.rustup.rs -sSf | sh
export PATH=$PATH:~/.cargo/bin
$CK_PYTHON -m pip uninstall h5py
$CK_PYTHON -m pip install h5py
</pre>


<a name="detect_python"></a>
## Detect Python

**NB:** Please detect only one Python interpreter. We recommend Python v3.8. While CK can normally detect available Python interpreters automatically, we are playing safe here by only detecting a particular one. Please only detect multiple Python interpreters, if you understand the consequences.

### <font color="#268BD0">Python v3.8</font>

<pre>
<b>[anton@dyson ~]&dollar;</b> ck detect soft:compiler.python --full_path=$(which python3.8)
<b>[anton@dyson ~]&dollar;</b> ck show env --tags=compiler,python
Env UID:         Target OS: Bits: Name:  Version: Tags:

b088ff37dc944f56   linux-64    64 python 3.8.12   64bits,compiler,host-os-linux-64,lang-python,python,target-os-linux-64,v3,v3.8,v3.8.12
</pre>

<a name="detect_gcc"></a>
## Detect (system) GCC

**NB:** CK can normally detect compilers automatically, but we are playing safe here.

<pre>
<b>[anton@dyson ~]&dollar;</b> which gcc
/opt/rh/devtoolset-10/root/usr/bin/gcc
<b>[anton@dyson ~]&dollar;</b> ck detect soft:compiler.gcc --full_path=$(which gcc)
<b>[anton@dyson ~]&dollar;</b> ck show env --tags=compiler,gcc
Env UID:         Target OS: Bits: Name:          Version: Tags:

fc44d4198510c275   linux-64    64 GNU C compiler 10.2.1   64bits,compiler,gcc,host-os-linux-64,lang-c,lang-cpp,target-os-linux-64,v10,v10.2,v10.2.1
</pre>

<a name="install_cmake"></a>
## Detect (system) CMake or install CMake from source

<a name="install_cmake_detect"></a>
### <font color="#268BD0"><b>Detect</b></font>

Try detecting CMake on your system:
<pre>
<b>[anton@dyson ~]&dollar;</b> ck detect soft --tags=tool,cmake
<b>[anton@dyson ~]&dollar;</b> ck show env --tags=cmake
Env UID:         Target OS: Bits: Name: Version: Tags:

4b6cb0f07e9fd005   linux-64    64 cmake 3.17.5   64bits,cmake,host-os-linux-64,target-os-linux-64,tool,v3,v3.17,v3.17.5
</pre>

<a name="install_cmake_install"></a>
### Install

If this fails, install CMake from source:

<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=tool,cmake,from.source
<b>[anton@dyson ~]&dollar;</b> ck show env --tags=tool,cmake,from.source
Env UID:         Target OS: Bits: Name: Version: Tags:

415293550c8e9de3   linux-64    64 cmake 3.20.5   64bits,cmake,compiled-by-gcc,compiled-by-gcc-9.3.0,host-os-linux-64,source,target-os-linux-64,tool,v3,v3.20,v3.20.5
</pre>

<a name="install_python_deps"></a>
## Install Python dependencies (in userspace)

#### Install implicit dependencies via pip

**NB:** These dependencies are _implicit_, i.e. CK will not try to satisfy them. If they are not installed, however, the workflow will fail.

<pre>
<b>[anton@dyson ~]&dollar;</b> &dollar; export CK_PYTHON=/usr/bin/python3
<b>[anton@dyson ~]&dollar;</b> &dollar;{CK_PYTHON} -m pip install --user --ignore-installed pip setuptools wheel
<b>[anton@dyson ~]&dollar;</b> &dollar;{CK_PYTHON} -m pip install --user wheel pyyaml testresources onnx-simplifier
<b>[anton@dyson ~]&dollar;</b> &dollar;{CK_PYTHON} -m pip install --user tokenization nvidia-pyindex
<b>[anton@dyson ~]&dollar;</b> &dollar;{CK_PYTHON} -m pip install --user onnx-graphsurgeon==0.3.11
</pre>

#### Install explicit dependencies via CK (also via `pip`, but register with CK at the same time)

**NB:** These dependencies are _explicit_, i.e. CK will try to satisfy them automatically. On a machine with multiple versions of Python, things can get messy, so we are playing safe here.

<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=python-package,cython
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=python-package,absl
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=python-package,opencv-python-headless
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=python-package,numpy
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=python-package,onnx --force_version=1.8.1
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=python-package,matplotlib
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=lib,python-package,pytorch --force_version=1.8.1
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=lib,python-package,transformers --force_version=2.4.0
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=lib,python-package,tensorflow
</pre>


<a name="install_inference_repo"></a>
## Install the MLPerf Inference repo and build LoadGen

<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=mlperf,inference,source
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=mlperf,loadgen,static
</pre>


<a name="prepare_squad"></a>
## Prepare the SQuAD validation dataset

<a name="prepare_squad_download"></a>
###  Download

<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package --ask --tags=dataset,squad,original,downloaded
</pre>


<a name="prepare_squad_preprocess"></a>
### Preprocess

<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package --ask \
--tags=dataset,tokenized,converted,raw
</pre>

<a name="prepare_bert_large"></a>
## Prepare the BERT Large model

<a name="prepare_squad_calibrate_preprocess"></a>
### Preprocess calibration dataset

<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package --ask \
--tags=dataset,tokenized,converted,pickle,calibration
</pre>


<a name="prepare_bert_large_calibrate"></a>
### Calibrate the model

<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package --tags=profile,bert-packed,qaic
</pre>


### Compile the Server/Offline model for the PCIe server cards

<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package \
--tags=model,qaic,bert
</pre>


### Compile and install the models to the 8 NSP AEDKs

#### Offline
<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package \
--tags=model,qaic,bert

<b>[anton@dyson ~]&dollar;</b> ck install package --tags=install-to-aedk \
--dep_add_tags.model-qaic=bert,model,compiled \
--env.CK_AEDK_IPS="aedk2" --env.CK_AEDK_PORTS="3232" --env.CK_AEDK_USER=$USER
</pre>

#### SingleStream
<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package \
--tags=model,qaic,bert

<b>[anton@dyson ~]&dollar;</b> ck install package --tags=install-to-aedk \
--dep_add_tags.model-qaic=bert,model,compiled \
--env.CK_AEDK_IPS="aedk2" --env.CK_AEDK_PORTS="3232" --env.CK_AEDK_USER=$USER
</pre>

### Compile and install the models to the 16 NSP AEDK

#### Offline
<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package \
--tags=model,qaic,bert

<b>[anton@dyson ~]&dollar;</b> ck install package --tags=install-to-aedk \
--dep_add_tags.model-qaic=bert,model,compiled \
--env.CK_AEDK_IPS="aedk3" --env.CK_AEDK_PORTS="3233" --env.CK_AEDK_USER=$USER
</pre>

#### SingleStream
<pre>
<b>[anton@dyson ~]&dollar;</b> ck install package \
--tags=model,qaic,bert

<b>[anton@dyson ~]&dollar;</b> ck install package --tags=install-to-aedk \
--dep_add_tags.model-qaic=bert,model,compiled \
--env.CK_AEDK_IPS="aedk3" --env.CK_AEDK_PORTS="3233" --env.CK_AEDK_USER=$USER
</pre>

<a name="benchmark"></a>
# Benchmark

- Offline: refer to [`README.offline.md`](https://github.com/krai/ck-qaic/blob/main/program/packed-bert-qaic-loadgen/README.offline.md).
- Server: refer to [`README.server.md`](https://github.com/krai/ck-qaic/blob/main/program/packed-bert-qaic-loadgen/README.server.md).
- Single Stream: refer to [`README.singlestream.md`](https://github.com/krai/ck-qaic/blob/main/program/packed-bert-qaic-loadgen/README.singlestream.md).

## Info

Please contact anton@krai.ai if you have any problems or questions.
