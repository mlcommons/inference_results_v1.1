Regarding GPUDirect memory access, there are two systems of Inspur involved, NF5488A5 and NF5688M6. All of them support the GPU Direct capability of NVIDIA GPUs to transfer data direct from PCIe devices directly to GPU device memory. 

# NF5488A5 system architecture
Each pair of A100(SXM) GPUs in NF5488A5 system is connected to a PCIe-Gen4 bridge, which is also connected to a Mellanox GDR NIC with bandwidth of 200 Gb/s. Inspur has measured over 11 GB/s per GPU when two GPUs that connected to the same NIC were under test concurrently. If only one GPU was under test, the bandwidth was over 22 GB/s. The highest bandwidth requirement per GPU for Inspur's submissions on NF5488A5 is about 14GB/s for the 3D-Unet. The bandwidth of the other submissions were well below 11 GB/s per GPU. 

# NF5688M6 system architecture
Every A100(SXM) GPUs in NF5688M6 system is connected to a PCIe-Gen4 bridge, which is also connected to a Mellanox GDR NIC with bandwidth of 200 Gb/s. Inspur has measured over 22 GB/s per GPU. The highest bandwidth requirement per GPU for Inspur's submissions on NF5688M6 is about 15GB/s for the 3D-Unet. 

