/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pinned_memory_pool.hpp"
#include "cuda_runtime_api.h"

// NUMA utils
#include "utils.hpp"

namespace triton_frontend {

PinnedMemoryPool::PinnedMemoryPool(const size_t element_count, const size_t element_byte_size)
    : m_Head(nullptr), m_Tail(nullptr), m_Blocks(element_count), m_Buffer(nullptr)
{
    FAIL_IF_CUDA_ERR(
        cudaHostAlloc(&m_Buffer, element_count * element_byte_size, cudaHostAllocPortable),
        "failed to allocate pinned memory");
    char* next_buffer = m_Buffer;
    for(auto& block : m_Blocks)
    {
        if(m_Tail == nullptr)
        {
            m_Tail = &block;
        }
        if(m_Head != nullptr)
        {
            block.m_NextBlock = m_Head;
        }
        m_Head = &block;
        block.m_Data = next_buffer;
        next_buffer += element_byte_size;
    }
}

PinnedMemoryPool::~PinnedMemoryPool()
{
    if(m_Buffer != nullptr)
    {
        FAIL_IF_CUDA_ERR(cudaFreeHost(m_Buffer), "failed to free pinned memory");
    }
}

PinnedMemoryPoolEnsemble::PinnedMemoryPoolEnsemble(
    const size_t per_instance_element_count, const size_t element_byte_size,
    const std::map<size_t, size_t>& gpu_instance_count, const NumaConfig& numa_config)
{
    size_t largest_gpu_idx = 0;
    for(const auto& gpu_instance : gpu_instance_count)
    {
        largest_gpu_idx = std::max(gpu_instance.first, largest_gpu_idx);
    }
    pools_ = std::vector<std::shared_ptr<PinnedMemoryPool>>(largest_gpu_idx + 1);

    if(numa_config.empty())
    {
        for(const auto& gpu_instance : gpu_instance_count)
        {
            // Use lockless version of the pool if we know that there will not
            // be shared access to the GPU specific pool
            if((gpu_instance.second > 1))
            {
                pools_[gpu_instance.first] = std::make_shared<MutexPinnedMemoryPool>(
                    per_instance_element_count * gpu_instance.second, element_byte_size);
            }
            else
            {
                pools_[gpu_instance.first] = std::make_shared<PinnedMemoryPool>(
                    per_instance_element_count * gpu_instance.second, element_byte_size);
            }
        }
    }
    else
    {
        auto gpu_numa_map = getGpuToNumaMap(numa_config);
        for(const auto& gpu_instance : gpu_instance_count)
        {
            bindNumaMemPolicy(gpu_numa_map[gpu_instance.first], numa_config.size());
            // Use lockless version of the pool if we know that there will not
            // be shared access to the GPU specific pool
            if((gpu_instance.second > 1))
            {
                pools_[gpu_instance.first] = std::make_shared<MutexPinnedMemoryPool>(
                    per_instance_element_count * gpu_instance.second, element_byte_size);
            }
            else
            {
                pools_[gpu_instance.first] = std::make_unique<PinnedMemoryPool>(
                    per_instance_element_count * gpu_instance.second, element_byte_size);
            }
            resetNumaMemPolicy();
        }
    }
}

} // namespace triton_frontend