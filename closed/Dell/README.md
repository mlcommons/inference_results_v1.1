# MLPerf Inference v1.1 Implementations
This is a repository of DellEMC servers using optimized implementations for [MLPerf Inference Benchmark v1.1](https://www.mlperf.org/inference-overview/).

# CPU Implementations
## Benchmarks
**Please refer to Intel's readme under /closed/Intel for detailed instructions, including performace guides, and instructions on how to run with new systems.**
  
The following benchmarks are part of our submission for MLPerf Inference v1.1:
- [3d-unet](/closed/Intel/code/3d-unet-99.9/openvino/README.md)
- [resnet50](/closed/Intel/code/resnet/openvino/README.md)
- [ssd-resnet34](/closed/Intel/code/ssd-resnet34/openvino/README.md)

## Dell Submission Systems

The closed systems that Dell has submitted on using CPUs are:
- Datacenter systems
  - Dell EMC PowerEdge R750 - Intel(R) Xeon(R) Gold 6330 CPU @ 2.0GHz
  - Dell EMC PowerEdge R750 - Intel(R) Xeon(R) Platinum 8352M CPU @ 2.30GHz


# GPU Implementations
## Benchmarks
**Please refer to /closed/NVIDIA for detailed instructions, including performace guides, and instructions on how to run with new systems.** 
  
The following benchmarks are part of our submission for MLPerf Inference v1.1:
- [3d-unet](code/3d-unet/tensorrt/README.md)
- [bert](code/bert/tensorrt/README.md)
- [dlrm](code/dlrm/tensorrt/README.md)
- [rnnt](code/rnnt/tensorrt/README.md)
- [resnet50](code/resnet50/tensorrt/README.md)
- [ssd-resnet34](code/ssd-resnet34/tensorrt/README.md)

# Dell Submission Systems

The closed systems that Dell EMC has submitted on using NVIDIA GPUs are:
- Datacenter systems
  - Dell EMC DSS 8440
    - A100-PCIe-80GB
    - A30
  - Dell EMC PowerEdge R750xa
    - A100-PCIe-40GB
    - A100-PCIe-80GB
    - A100-PCIe-80GB - 28x MIG 1g.10gb
  - Dell EMC PowerEdge R7525
    - A100-PCIe-40GB
    - A30
  - Dell EMC PowerEdge XE2420
    - A10
  - Dell EMC PowerEdge XE8545
    - A100-SXM-80GB / 500W
    - A100-SXM-80GB / 500W - 28x MIG 1g.10gb
  - Dell EMC PowerEdge XR12
    - A10
- Edge systems
  - Dell EMC PowerEdge XE2420
    - A10
  - Dell EMC PowerEdge XR12
    - A10

#VMware Submission Systems
The closed systems that VMware has submitted on in partnership with Dell using NVIDIA vGPUs are:
- Datacenter systems
  - Dell EMC PowerEdge R7525
    - GRID vGPU A100-PCIe-40C
