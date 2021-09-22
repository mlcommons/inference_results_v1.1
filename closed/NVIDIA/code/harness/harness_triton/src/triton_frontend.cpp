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

#include "triton_frontend.hpp"

// Triton
#include "model_config.pb.h"
#include "src/core/constants.h"
#include "src/servers/common.h"

// Google Logging
#include <glog/logging.h>

// General C++
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

// LoadGen
#include "loadgen.h"

// DLRM QSL
#include "dlrm_qsl.hpp"

/* Use Triton namespace */
namespace ni = nvidia::inferenceserver;

/* Define macro for enabling tracing */
#ifndef TRITON_FRONTEND_TRACE
#define TRITON_FRONTEND_TRACE 0
#endif

namespace triton_frontend {

void Server_SUT::Init(size_t min_sample_size, size_t max_sample_size,
                      size_t buffer_manager_thread_count, bool batch_triton_requests,
                      bool check_contiguity, const std::string& numa_config_str)
{
    Triton_Server_SUT::Init(min_sample_size, max_sample_size, buffer_manager_thread_count,
                            batch_triton_requests, check_contiguity, numa_config_str);

    size_t max_batch1_byte_size = 0;
    std::vector<std::string> output_names;
    m_MaxBatchSize = m_Config.max_batch_size() == 0 ? 1 : m_Config.max_batch_size();
    // Pre-allocate a memory pool for output buffers
    // Use the largest possible size among the outputs as size for each block
    for(const auto& output : m_Config.output())
    {
        int batch1_byte_size = 1;
        for(const auto& dim : output.dims())
        {
            // FIXME: hard-coded value for variable dims
            if (dim == -1)
            {
                batch1_byte_size *= BERT_MAX_SEQ_LENGTH;
            }
            else
            {
                batch1_byte_size *= dim;
            }
        }
        batch1_byte_size *= GetDataTypeByteSize(output.data_type());
        if (batch1_byte_size <= 0)
        {
            FAIL("can't preallocate memory for variable size data type");
        }
        max_batch1_byte_size = std::max(max_batch1_byte_size, (size_t) batch1_byte_size);
        output_names.emplace_back(output.name());
    }

    int max_batchn_byte_size = max_batch1_byte_size * m_MaxBatchSize;
    std::map<size_t, size_t> gpu_instance_count;
    for(const auto& instance_group : m_Config.instance_group())
    {
        for(const auto& gpu : instance_group.gpus())
        {
            auto it = gpu_instance_count.find(gpu);
            if (it == gpu_instance_count.end()) {
                gpu_instance_count[gpu] = instance_group.count();
            } else {
                it->second += instance_group.count();
            }
        }
    }

    size_t pool_item_count_per_instance =
#ifdef __aarch64__
        2 * (m_MaxBatchSize / min_sample_size + 1) * output_names.size();
#else
        4 * (m_MaxBatchSize / min_sample_size + 1) * output_names.size();
#endif
    size_t pool_item_size = max_batch1_byte_size * max_sample_size;

    auto numa_config = parseNumaConfig(numa_config_str);
    auto output_mem_type = m_EndOnDevice ? TRITONSERVER_MEMORY_GPU : TRITONSERVER_MEMORY_CPU;
    m_OutputBufferPool.reset(new PinnedMemoryPoolEnsemble(pool_item_count_per_instance,
                                                          pool_item_size, gpu_instance_count,
                                                          numa_config, output_mem_type));

    if(m_EndOnDevice)
    {
        m_OutputBufferPoolEndOnDevice.reset(
            new PinnedMemoryPoolEnsemble(pool_item_count_per_instance, pool_item_size,
                                         gpu_instance_count, numa_config, TRITONSERVER_MEMORY_CPU));
    }

    LOG(INFO) << "Allocated Pinned memory pool of " << pool_item_count_per_instance * pool_item_size
              << " bytes for every instance.";

    // Set the host policy names that maps to each GPU device,
    // here uses the default host policy name Triton would generate for
    // model instances on the corresponding device.
    for(size_t idx = 0; idx < gpu_instance_count.size(); ++idx)
    {
        m_HostPolicyNames.emplace_back("gpu_" + std::to_string(idx));
    }
    m_NumGPUs = gpu_instance_count.size();

    // For batched requests, we need 2 sets of output buffers, one of them is
    // used
    // for the
    // batched requests and the other for single-sample requests. The size and
    // numbers of these
    // buffers are different.
    if (m_BatchTritonRequests)
    {
        size_t pool_item_size_batch = max_batchn_byte_size;
        // FIXME Madhu - need a different count for server and offline
        size_t pool_item_count_batch = 4 * output_names.size();
        m_BatchedOutputBufferPool.reset(
            new PinnedMemoryPoolEnsemble(pool_item_count_batch, pool_item_size_batch,
                                         gpu_instance_count, numa_config, output_mem_type));
        if(m_EndOnDevice)
        {
            m_BatchedOutputBufferPoolEndOnDevice.reset(new PinnedMemoryPoolEnsemble(
                pool_item_count_batch, pool_item_size_batch, gpu_instance_count, numa_config,
                TRITONSERVER_MEMORY_CPU));
        }
        LOG(INFO) << "Allocated Pinned memory pool of count: " << pool_item_count_batch
                  << " size :" << pool_item_size_batch << " bytes for batched requests.";
    }

    // Pre-allocate a growable request pool for inference requests
    std::vector<InputMetaData> inputs;
    m_IsDynamic = false;
    for(const auto& io : m_Config.input())
    {
        InputMetaData input;
        std::get<0>(input) = io.name();
        std::get<1>(input) = DataTypeToTriton(io.data_type());
        auto& shape = std::get<2>(input);
        if(m_Config.max_batch_size() != 0)
        {
            shape.push_back(m_BatchTritonRequests ? m_MaxBatchSize : 1);
        }
        for (const auto& dim : io.dims())
        {
            m_IsDynamic |= (dim == -1);
            shape.push_back(dim);
        }
        m_InputTensors.emplace_back(input);
        inputs.emplace_back(std::move(input));
    }

    /*  Create the allocator that will be used to allocate buffers for
        the result tensors. */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&m_Allocator, ResponseAlloc, ResponseRelease,
                                                  nullptr /* start_fn */),
                "creating response allocator");

    RequestPool::Create(m_RequestCount /* initial_element_count */, m_Server.get(), this,
                        m_ModelName, m_ModelVersion, inputs, output_names);

    // Prepare padding buffer in the case of DLRM. The model assumes
    // even batch size but some sample has odd batch size
    if(m_UseDlrmQsl)
    {
        for(const auto& io : m_Config.output())
        {
            int64_t batch1_byte_size =
                TRITONSERVER_DataTypeByteSize(DataTypeToTriton(io.data_type()));
            for(const auto& dim : io.dims())
            {
                batch1_byte_size *= dim;
            }
            m_OutputPaddingSize = (size_t) batch1_byte_size;
        }
    }

    /* Set the number of warmup responses to 0 to prepare for next warmup */
    m_NumWarmupResponses = 0;
}

void Server_SUT::Warmup(double duration_sec, double expected_qps)
{
    /* Notify user that we are starting the warmup */
    LOG(INFO) << "Starting Triton warmup" << std::endl;

    /* Calculate the number of inferences to send
       An "inference" can either be a single sample or a batch, depending on BatchTritonRequests.
       We should scale our num_inferences appropriately.
    */
    auto num_inferences = static_cast<int>((duration_sec * expected_qps) /
                                           (m_BatchTritonRequests ? m_MaxBatchSize : 1));

    /* Keep track of the number of inferences that we have sent so far */
    int inferences_sent = 0;

    // Load a sample to RAM to use
    mlperf::QuerySampleIndex index{0}; // Arbitrary sample index
    std::vector<mlperf::QuerySampleIndex> samples;
    samples.push_back(index);
    for (auto& qsl : m_SampleLibraries) {
        qsl->LoadSamplesToRam(samples);
    }

    while (inferences_sent < num_inferences)
    {
        /* Create the inference request provider, which provides the request
            header information as well as the actual data. */
        auto request_block = RequestPool::Obtain(0);
        if (m_UseDlrmQsl)
        {
            // Inputs will need to be re-added as the shape is different from run
            // to run
            TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
            // new batch size for the request
            auto num_pairs = qsl->GetNumUserItemPairs(index);

            // Set default input buffer
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*) qsl->GetSampleAddress(index, idx); // Get address of the query
                size_t single_sample_size = qsl->GetSampleSize(idx);

                auto& shape = std::get<2>(m_InputTensors[idx]);
                shape[0] = num_pairs;
                if (num_pairs % 2)
                {
                    shape[0] += 1;
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), std::get<1>(m_InputTensors[idx]),
                                    shape.data(), shape.size()),
                        "re-adding input");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size * num_pairs, m_InputMemoryType,
                                    0),
                                "appending input data");
                    // Add padding buffer
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size, m_InputMemoryType, 0),
                                "appending input data padding");

                    if (m_StartFromDevice && m_NumGPUs > 1) {
                      for (int i = 0; i < m_NumGPUs; i++) {
                          input_data = (int8_t*)qsl->GetSampleAddress(index, idx, i);

                          FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                          request_block->m_Data,
                                          std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                          single_sample_size * num_pairs, m_InputMemoryType, i,
                                          m_HostPolicyNames[i].c_str()),
                                      "appending input data with host policy");

                          FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                          request_block->m_Data,
                                          std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                          single_sample_size, m_InputMemoryType, i,
                                          m_HostPolicyNames[i].c_str()),
                                      "appending input data padding");
                      }
                    }
                }
                else
                {
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                                "re-adding input");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size * num_pairs, m_InputMemoryType,
                                    0),
                                "appending input data");

                    if(m_StartFromDevice && m_NumGPUs > 1)
                    {
                        for(int i = 0; i < m_NumGPUs; i++)
                        {
                            input_data = (int8_t*)qsl->GetSampleAddress(
                                index, idx, i); // Get address of the query for device

                            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                            request_block->m_Data,
                                            std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                            single_sample_size * num_pairs, m_InputMemoryType, i,
                                            m_HostPolicyNames[i].c_str()),
                                        "appending input data with host policy");
                        }
                    }
                }
            }

            // If there are more than one QSL, we want to add input buffer with device affinity,
            // use the proper host policy name recognized by Triton.
            if (m_SampleLibraries.size() > 1)
            {
                for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
                {
                    auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[qsl_idx].get());
                    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
                    {
                        // Get a pointer to the input data
                        int8_t* input_data = (int8_t*) qsl->GetSampleAddress(index, idx); // Get address of the query
                        size_t single_sample_size = qsl->GetSampleSize(idx);

                        if (num_pairs % 2)
                        {
                            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                            std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                            single_sample_size * num_pairs, m_InputMemoryType, 0,
                                            m_HostPolicyNames[qsl_idx].c_str()),
                                "appending input data");
                            // Add padding buffer
                            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size,
                                            m_InputMemoryType, 0, m_HostPolicyNames[qsl_idx].c_str()),
                                "appending input data padding");
                        }
                        else
                        {
                            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                            std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                            single_sample_size * num_pairs, m_InputMemoryType, 0,
                                            m_HostPolicyNames[qsl_idx].c_str()),
                                "appending input data");
                        }
                    }
                }
            }
        }
        else if (m_IsDynamic)
        {
            // Special handling as BERT is the only model uses dynamic shape
            //
            // Inputs will need to be re-added as the shape is different from run
            // to run
            TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

            size_t seq_len = GetSampleLength(m_SampleLibraries[0], index);
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data
                    = (int8_t*) m_SampleLibraries[0]->GetSampleAddress(index, idx); // Get address of the query
                // Need to calculate the shape from data for dynamic case
                size_t input_size = seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

                thread_local std::vector<int64_t> shape{1, 0};
                shape[1] = seq_len;
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), std::get<1>(m_InputTensors[idx]),
                                shape.data(), shape.size()),
                    "re-adding input");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
                    "appending input data");
            }

            // If there are more than one QSL, we want to add input buffer with device affinity,
            // use the proper host policy name recognized by Triton.
            if (m_SampleLibraries.size() > 1)
            {
                for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
                {
                    auto qsl = m_SampleLibraries[qsl_idx].get();

                    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
                    {
                        // Get a pointer to the input data
                        int8_t* input_data
                            = (int8_t*) qsl->GetSampleAddress(index, idx); // Get address of the query
                        // Need to calculate the shape from data for dynamic case
                        size_t input_size = seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

                        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                        std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0,
                                        m_HostPolicyNames[qsl_idx].c_str()),
                            "appending input data");
                    }
                }
            }
        }
        else
        {
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*)m_SampleLibraries[0]->GetSampleAddress(
                    index, idx); // Get address of the query
                size_t input_size = m_SampleLibraries[0]->GetSampleSize(idx) *
                                    (m_BatchTritonRequests ? m_MaxBatchSize : 1);

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
                            "removing input data");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
                    "appending input data");
            }

            // If there are more than one QSL, we want to add input buffer with device affinity,
            // use the proper host policy name recognized by Triton.
            if (m_SampleLibraries.size() > 1)
            {
                for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
                {
                    auto qsl = m_SampleLibraries[qsl_idx].get();

                    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
                    {
                        // Get a pointer to the input data
                        int8_t* input_data = (int8_t*)qsl->GetSampleAddress(
                            index, idx); // Get address of the query
                        size_t input_size = qsl->GetSampleSize(idx) *
                                            (m_BatchTritonRequests ? m_MaxBatchSize : 1);

                        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0,
                                m_HostPolicyNames[qsl_idx].c_str()),
                            "appending input data");
                    }
                }
            }
        }


        /* Set response callback for warmup */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                        request_block->m_Data, m_Allocator, m_OutputBufferPool.get(), WarmupResponseComplete, this),
            "appending input data");

        /* Actually perform inferences (asynchronously) */
        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, nullptr), "running inference");
        inferences_sent += 1;
    }

    /* Wait for all the warmup inferences to complete */
    while (m_NumWarmupResponses < inferences_sent)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    /* Unload sample from RAM */
    for (auto& qsl : m_SampleLibraries) {
        qsl->UnloadSamplesFromRam(samples);
    }

    /* Reset the number of warmup responses */
    m_NumWarmupResponses = 0;

    /* Notify user that we are done with the warmup */
    LOG(INFO) << "Finished Triton warmup" << std::endl;
}

void Server_SUT::HandleSingleQuery(const std::vector<mlperf::QuerySample>& samples,
                                   int indexIntoQuerySample, int pool_idx)
{
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE

    auto request_block = RequestPool::Obtain(0);
    request_block->m_ResponseMetadata.m_ResponseId = samples[indexIntoQuerySample].id;
    request_block->m_ResponseMetadata.m_QuerySampleIdx = samples[indexIntoQuerySample].index;
    request_block->m_ResponseMetadata.m_TracePtr = trace;
    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*) m_SampleLibraries[0]->GetSampleAddress(
            samples[indexIntoQuerySample].index, idx); // Get address of the query
        size_t input_size = m_SampleLibraries[0]->GetSampleSize(idx);

        FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                        request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
            "removing input data");
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                        request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                        input_size, m_InputMemoryType, 0),
                    "appending input data");

        if(m_StartFromDevice && m_NumGPUs > 1)
        {
            for(int i = 0; i < m_NumGPUs; i++)
            {
                input_data = (int8_t*)m_SampleLibraries[0]->GetSampleAddress(
                    samples[indexIntoQuerySample].index, idx,
                    i); // Get address of the query for device

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                input_data, input_size, m_InputMemoryType, i,
                                m_HostPolicyNames[i].c_str()),
                            "appending input data with host policy");
            }
        }
    }

    // If there are more than one QSL, we want to add input buffer with device affinity,
    // use the proper host policy name recognized by Triton.
    if (m_SampleLibraries.size() > 1)
    {
        for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
        {
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*) m_SampleLibraries[qsl_idx]->GetSampleAddress(
                    samples[indexIntoQuerySample].index, idx); // Get address of the query
                size_t input_size = m_SampleLibraries[qsl_idx]->GetSampleSize(idx);

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                        std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0,
                        m_HostPolicyNames[qsl_idx].c_str()),
                    "appending input data");
            }
        }
    }

    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
        "appending input data");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

void Server_SUT::HandleSingleBertQuery(
    const std::vector<mlperf::QuerySample>& samples, int indexIntoQuerySample, int pool_idx)
{
    // If its bert used samples in the sorted order
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE
       // Set the Request Provider
    auto request_block = RequestPool::Obtain(pool_idx);

    request_block->m_ResponseMetadata.m_ResponseId = samples[indexIntoQuerySample].id;
    request_block->m_ResponseMetadata.m_QuerySampleIdx = samples[indexIntoQuerySample].index;
    request_block->m_ResponseMetadata.m_TracePtr = trace;

    // Special handling as BERT is the only model uses dynamic shape
    //
    // Inputs will need to be re-added as the shape is different from run
    // to run
    TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

    size_t seq_len = GetSampleLength(m_SampleLibraries[0], samples[indexIntoQuerySample].index);
    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*) m_SampleLibraries[0]->GetSampleAddress(
            samples[indexIntoQuerySample].index, idx); // Get address of the query
        // Need to calculate the shape from data for dynamic case
        size_t input_size = seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

        thread_local std::vector<int64_t> shape{1, 0};
        shape[1] = seq_len;
        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
            "re-adding input");
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                        request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                        input_size, m_InputMemoryType, 0),
                    "appending input data");
        if(m_StartFromDevice && m_NumGPUs > 1)
        {
            for(int i = 0; i < m_NumGPUs; i++)
            {
                input_data = (int8_t*)m_SampleLibraries[0]->GetSampleAddress(
                    samples[indexIntoQuerySample].index, idx,
                    i); // Get address of the query for device

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                input_data, input_size, m_InputMemoryType, i,
                                m_HostPolicyNames[i].c_str()),
                            "appending input data with host policy");
            }
        }
    }

    // If there are more than one QSL, we want to add input buffer with device affinity,
    // use the proper host policy name recognized by Triton.
    if (m_SampleLibraries.size() > 1)
    {
        for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
        {
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*) m_SampleLibraries[qsl_idx]->GetSampleAddress(
                    samples[indexIntoQuerySample].index, idx); // Get address of the query
                // Need to calculate the shape from data for dynamic case
                size_t input_size = seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                        std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0,
                        m_HostPolicyNames[qsl_idx].c_str()),
                    "appending input data");
            }
        }
    }

    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
        "appending input data");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

void Server_SUT::HandleSingleDlrmQuery(
    const std::vector<mlperf::QuerySample>& samples, int indexIntoQuerySample, int pool_idx)
{
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE
       // Set the Request Provider
    auto request_block = RequestPool::Obtain(pool_idx);

    request_block->m_ResponseMetadata.m_ResponseId = samples[indexIntoQuerySample].id;
    request_block->m_ResponseMetadata.m_QuerySampleIdx = samples[indexIntoQuerySample].index;
    request_block->m_ResponseMetadata.m_TracePtr = trace;
    // Inputs will need to be re-added as the shape is different from run
    // to run
    TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

    auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
    // new batch size for the request
    auto num_pairs = qsl->GetNumUserItemPairs(samples[indexIntoQuerySample].index);
    auto l_InputTensors = m_InputTensors;

    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*) qsl->GetSampleAddress(samples[indexIntoQuerySample].index,
            idx); // Get address of the query
        const size_t single_sample_size = qsl->GetSampleSize(idx);

        auto& shape = std::get<2>(l_InputTensors[idx]);
        shape[0] = num_pairs;
        if (num_pairs % 2)
        {
            shape[0] += 1;
            FAIL_IF_ERR(
                TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                "re-adding input");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size * num_pairs,
                            m_InputMemoryType, 0),
                "appending input data");
            // Add padding buffer
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                            input_data, single_sample_size, m_InputMemoryType, 0),
                        "appending input data padding");

            if(m_StartFromDevice && m_NumGPUs > 1)
            {
                for(int i = 0; i < m_NumGPUs; i++)
                {
                    input_data =
                        (int8_t*)qsl->GetSampleAddress(samples[indexIntoQuerySample].index, idx, 0,
                                                       i); // Get address of the query for device
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size * num_pairs, m_InputMemoryType,
                                    i, m_HostPolicyNames[i].c_str()),
                                "appending input data");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size, m_InputMemoryType, i,
                                    m_HostPolicyNames[i].c_str()),
                                "appending input data with host policy");
                }
            }
        }
        else
        {
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                            std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                        "re-adding input");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                            input_data, single_sample_size * num_pairs, m_InputMemoryType, 0),
                        "appending input data");
            if(m_StartFromDevice && m_NumGPUs > 1)
            {
                for(int i = 0; i < m_NumGPUs; i++)
                {
                    input_data =
                        (int8_t*)qsl->GetSampleAddress(samples[indexIntoQuerySample].index, idx, 0,
                                                       i); // Get address of the query for device

                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size * num_pairs, m_InputMemoryType,
                                    i, m_HostPolicyNames[i].c_str()),
                                "appending input data with host policy");
                }
            }
        }
    }

    // If there are more than one QSL, we want to add input buffer with device affinity,
    // use the proper host policy name recognized by Triton.
    if (m_SampleLibraries.size() > 1)
    {
        for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
        {
            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[qsl_idx].get());
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*) qsl->GetSampleAddress(samples[indexIntoQuerySample].index,
                    idx); // Get address of the query
                const size_t single_sample_size = qsl->GetSampleSize(idx);

                if (num_pairs % 2)
                {
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size * num_pairs,
                                    m_InputMemoryType, 0, m_HostPolicyNames[qsl_idx].c_str()),
                        "appending input data");
                    // Add padding buffer
                    FAIL_IF_ERR(
                        TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size, m_InputMemoryType, 0,
                            m_HostPolicyNames[qsl_idx].c_str()),
                        "appending input data padding");
                }
                else
                {
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size * num_pairs,
                                    m_InputMemoryType, 0, m_HostPolicyNames[qsl_idx].c_str()),
                        "appending input data");
                }
            }
        }
    }

    request_block->m_ResponseMetadata.m_PaddingSize = (num_pairs % 2) ? m_OutputPaddingSize : 0;

    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
        "appending input data");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

bool Server_SUT::CheckContiguity(const std::vector<mlperf::QuerySample>& samples, int start_id, int end_id)
{
    // [FIXME] is it safe to assume if result from one QSL can apply to all other QSLs
    bool contiguous = true;

    for (size_t i = 0; i < m_InputTensors.size() && contiguous; i++)
    {
        auto prev = static_cast<int8_t*>(m_SampleLibraries[0]->GetSampleAddress(samples[start_id].index, i));

        auto num_pairs = 0;
        auto sample_size = m_SampleLibraries[0]->GetSampleSize(i);

        for (auto it = start_id + 1; it < end_id; ++it)
        {
            auto next = static_cast<int8_t*>(m_SampleLibraries[0]->GetSampleAddress(samples[it].index, i));

            if (next != prev + sample_size)
            {
                contiguous = false;
                break;
            }
            prev = next;
        }
    }
    return contiguous;
}

bool Server_SUT::CheckDLRMContiguity(
    const std::vector<mlperf::QuerySample>& samples, int start_id, int end_id, int num_pairs)
{
    // [FIXME] is it safe to assume if result from one QSL can apply to all other QSLs
    auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());

    for (size_t i = 0; i < m_InputTensors.size(); i++)
    {
        int8_t* first_task_start = static_cast<int8_t*>(qsl->GetSampleAddress(samples[start_id].index, i));

        int8_t* last_task_start = static_cast<int8_t*>(qsl->GetSampleAddress(samples[end_id - 1].index, i));

        auto num_pairs_last_task = qsl->GetNumUserItemPairs(samples[end_id - 1].index);
        auto single_sample_size = qsl->GetSampleSize(i);

        // new batch size for the request
        auto sample_size = single_sample_size * num_pairs;

        if (first_task_start + sample_size != last_task_start + (num_pairs_last_task * single_sample_size))
            return false;
    }
    return true;
}

void Server_SUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    if (m_BatchTritonRequests)
    {
        // Handle non-DLRM, non-BERT batching
        if (!m_UseDlrmQsl && !m_IsDynamic)
        {
            int num_batches = samples.size() / m_MaxBatchSize;
            int start_query_id = 0;
            int end_batch_id = 0;
            for (int i = 0; i < num_batches; i++)
            {
                end_batch_id = start_query_id + m_MaxBatchSize;
                bool contiguous = CheckContiguity(samples, start_query_id, end_batch_id);
                if (contiguous)
                {
                    IssueTritonContiguousBatch(samples, start_query_id, end_batch_id);
                }
                else
                {
                    IssueQueryInternal(samples, start_query_id, end_batch_id);
                }
                start_query_id = end_batch_id;
            }
            // Handle the spillover queries by sending individual queries
            if (start_query_id < samples.size() - 1)
            {
                IssueQueryInternal(samples, start_query_id, samples.size());
            }
            return;
        }
        // Handle DLRM batching
        if (m_UseDlrmQsl)
        {
            HandleDlrmBatchedQueries(samples);
            return;
        }
    }

    // Currently only DLRM will suffer from the IssueQuery bottleneck,
    // so always use the current thread for other benchmarks. Leaving this
    // optimization here in case we are not able to turn on batching requests
    // for all offline cases.
    if (!m_UseDlrmQsl || samples.size() < 100)
    {
        IssueQueryInternal(samples, 0, samples.size());
    }
    else
    {
        auto a1 = std::async(std::launch::async, &Server_SUT::IssueQueryInternal, this, samples, 0, samples.size() / 2);
        auto a2 = std::async(
            std::launch::async, &Server_SUT::IssueQueryInternal, this, samples, samples.size() / 2, samples.size());
        a1.get();
        a2.get();
    }
}

void Server_SUT::HandleDlrmBatchedQueries(const std::vector<mlperf::QuerySample>& samples)
{
    // Offline
    if (m_CheckContiguity)
    {
        // For DLRM the batching will be different because we need to take account of
        // how many user item pairs are in a given batch rather than number of samples.
        int num_user_item_pairs = 0;
        int batched_samples = 0;
        int batch_start_sample = 0;
        int batch_end_sample = 0;
        auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
        while(batched_samples < samples.size())
        {
            // Find number of samples that will create batch
            num_user_item_pairs = 0;
            std::vector<size_t> batchNumPairs;
            for(int i = batch_start_sample;
                num_user_item_pairs < m_MaxBatchSize && i < samples.size(); i++)
            {
                auto num_pairs = qsl->GetNumUserItemPairs(samples[i].index);

                if(num_user_item_pairs + num_pairs >= m_MaxBatchSize || i == samples.size())
                {
                    batch_end_sample = i;
                    break;
                }
                num_user_item_pairs += num_pairs;
                batchNumPairs.push_back(num_pairs);
            }
            // Check if batch is contiguous
            bool contiguous = CheckDLRMContiguity(samples, batch_start_sample, batch_end_sample,
                                                  num_user_item_pairs);

            // Accumulate batch
            if(contiguous)
            {
                IssueContiguousBatchDLRMQuery(samples, batch_start_sample, batch_end_sample,
                                              num_user_item_pairs, batchNumPairs);
            }
            else
            {
                IssueQueryInternal(samples, batch_start_sample, batch_end_sample);
            }
            batched_samples += (batch_end_sample - batch_start_sample);
            batch_start_sample = batch_end_sample;
            batch_end_sample = batch_start_sample + 1;
        }
        return;
    }
    else // server
    {
        HandleDlrmServerQuery(samples);
    }
    return;
}

void Server_SUT::HandleDlrmServerQuery(const std::vector<mlperf::QuerySample>& samples)
{
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "DlrmBatchedQuery start");
    }
#endif // TRITON_FRONTEND_TRACE

    int num_user_item_pairs = 0;
    int batched_samples = 0;
    int batch_start_sample = 0;
    int batch_end_sample = 1;
    auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
    while (batched_samples < samples.size())
    {
        // Find number of samples that will create batch
        std::vector<size_t> batchNumPairs;
        auto num_pairs_first = qsl->GetNumUserItemPairs(samples[batch_start_sample].index);
        batchNumPairs.clear();
        // If there is a request being staged, don't wait to form batch
        if (m_CurrentRequest != nullptr
            && m_CurrentRequest->m_BatchResponseMetadata.m_DlrmTotalNumPairs + num_pairs_first > m_MaxBatchSize)
        {
            IssueCurrentBatchedRequest(m_CurrentRequest, trace);
        }
        auto num_waiting_pairs
            = m_CurrentRequest == nullptr ? 0 : m_CurrentRequest->m_BatchResponseMetadata.m_DlrmTotalNumPairs;
        num_user_item_pairs = num_waiting_pairs + num_pairs_first;
        batchNumPairs.push_back(num_pairs_first);
        auto current_block_user_item_pairs = num_pairs_first;

        for (int i = batch_start_sample + 1; num_user_item_pairs < m_MaxBatchSize && i < samples.size(); i++)
        {
            auto num_pairs = qsl->GetNumUserItemPairs(samples[i].index);
            batch_end_sample = i;
            if (num_user_item_pairs + num_pairs > m_MaxBatchSize)
            {
                break;
            }

            num_user_item_pairs += num_pairs;
            current_block_user_item_pairs += num_pairs;
            batchNumPairs.push_back(num_pairs);
        }
        IssueOrQueueCoalescedDLRMQuery(
            samples, batch_start_sample, batch_end_sample, current_block_user_item_pairs, batchNumPairs);

        batched_samples += (batch_end_sample - batch_start_sample);
        batch_start_sample = batch_end_sample;
        batch_end_sample = batch_start_sample + 1;
        batchNumPairs.clear();
    }
}

// For server case // many queries but not contiguous
void Server_SUT::IssueOrQueueCoalescedDLRMQuery(const std::vector<mlperf::QuerySample>& samples,
    size_t batch_start_sample, size_t batch_end_sample, size_t num_user_item_pairs,
    const std::vector<size_t>& batchNumPairs)
{
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE
    RequestPool::Block* request_block;
    // Starting new request set
    if (m_CurrentRequest == nullptr)
    {
        request_block = RequestPool::Obtain(0);
        m_CurrentRequest = request_block;
        request_block->m_BatchResponseMetadata.m_TracePtr = trace;
        request_block->m_BatchResponseMetadata.m_DlrmNumPairsList.clear();
        request_block->m_BatchResponseMetadata.m_ResponseId.clear();
        request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.clear();
        request_block->m_BatchResponseMetadata.m_DlrmTotalNumPairs = num_user_item_pairs;
        request_block->m_BatchResponseMetadata.m_RequestBatchSize = m_MaxBatchSize;
    }
    else // Queue this query on the sample request block or "Issue" if max_size
         // is exceeded
    {
        request_block = m_CurrentRequest;
        request_block->m_BatchResponseMetadata.m_DlrmTotalNumPairs += num_user_item_pairs;
        request_block->m_BatchResponseMetadata.m_RequestBatchSize = m_MaxBatchSize;
    }

    for(size_t i = batch_start_sample; i < batch_end_sample; i++)
    {
        request_block->m_BatchResponseMetadata.m_ResponseId.push_back(samples[i].id);
        request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.push_back(samples[i].index);
        request_block->m_BatchResponseMetadata.m_DlrmNumPairsList.push_back(
            batchNumPairs[i - batch_start_sample]);
        request_block->m_BatchResponseMetadata.m_RequestBatchSize = m_MaxBatchSize;
        m_QueuedSamples.push_back(samples[i]);
    }

    // Issue query if adding current request has created a full batch size
    if (request_block->m_BatchResponseMetadata.m_DlrmTotalNumPairs == m_MaxBatchSize)
    {
        IssueCurrentBatchedRequest(request_block, trace);
    }
}

void Server_SUT::IssueCurrentBatchedRequest(RequestPool::Block* request_block, TRITONSERVER_InferenceTrace* trace)
{
    size_t totalNumberOfUserItemPairs = request_block->m_BatchResponseMetadata.m_DlrmTotalNumPairs;
    auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
    auto l_InputTensors = m_InputTensors;

    TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

    for (size_t i = 0; i < m_QueuedSamples.size(); i++)
    {
        for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
        {
            const size_t single_sample_size = qsl->GetSampleSize(idx);
            auto& shape = std::get<2>(l_InputTensors[idx]);
            shape[0] = totalNumberOfUserItemPairs;
            if (totalNumberOfUserItemPairs % 2)
            {
                shape[0] += 1;
            }
            int8_t* input_data = (int8_t*) qsl->GetSampleAddress(m_QueuedSamples[i].index,
                idx); // Get address of the query

            if (i == 0)
            {
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), std::get<1>(m_InputTensors[idx]),
                                shape.data(), shape.size()),
                    "re-adding input");
            }
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                            input_data,
                            single_sample_size *
                                request_block->m_BatchResponseMetadata.m_DlrmNumPairsList[i],
                            m_InputMemoryType, 0),
                        "appending input data");
            if(m_StartFromDevice && m_NumGPUs > 1)
            {
                for(int d = 0; d < m_NumGPUs; d++)
                {
                    auto input_data_device = (int8_t*)qsl->GetSampleAddress(
                        m_QueuedSamples[i].index, idx, 0, d); // Get address of the query for device

                    FAIL_IF_ERR(
                        TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                            input_data_device,
                            single_sample_size *
                                request_block->m_BatchResponseMetadata.m_DlrmNumPairsList[i],
                            m_InputMemoryType, d, m_HostPolicyNames[d].c_str()),
                        "appending input data");
                }
            }
            // Add one extra user item pair at the end if required
            if(i == m_QueuedSamples.size() - 1 && totalNumberOfUserItemPairs % 2)
            {
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                input_data, single_sample_size, m_InputMemoryType, 0),
                            "Appending padding input");

                if(m_StartFromDevice && m_NumGPUs > 1)
                {
                    for(int d = 0; d < m_NumGPUs; d++)
                    {
                        auto input_data_device = (int8_t*)qsl->GetSampleAddress(
                            m_QueuedSamples[i].index, idx, 0,
                            d); // Get address of the query for device

                        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                        request_block->m_Data,
                                        std::get<0>(m_InputTensors[idx]).c_str(), input_data_device,
                                        single_sample_size, m_InputMemoryType, d,
                                        m_HostPolicyNames[d].c_str()),
                                    "appending input data with host policy");
                    }
                }
            }
        }
    }

    // If there are more than one QSL, we want to add input buffer with device affinity,
    // use the proper host policy name recognized by Triton.
    if (m_SampleLibraries.size() > 1)
    {
        for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
        {
            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[qsl_idx].get());
            for (size_t i = 0; i < m_QueuedSamples.size(); i++)
            {
                for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
                {
                    const size_t single_sample_size = qsl->GetSampleSize(idx);
                    int8_t* input_data = (int8_t*) qsl->GetSampleAddress(m_QueuedSamples[i].index,
                        idx); // Get address of the query
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                    single_sample_size * request_block->m_BatchResponseMetadata.m_DlrmNumPairsList[i],
                                    m_InputMemoryType, 0, m_HostPolicyNames[qsl_idx].c_str()),
                        "appending input data");
                    // Add one extra user item pair at the end if required
                    if (i == m_QueuedSamples.size() - 1 && totalNumberOfUserItemPairs % 2)
                    {
                        FAIL_IF_ERR(
                            TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size, m_InputMemoryType, 0,
                                m_HostPolicyNames[qsl_idx].c_str()),
                            "Appending padding input");
                    }
                }
            }
        }
    }

    request_block->m_BatchResponseMetadata.m_PaddingSize = (totalNumberOfUserItemPairs % 2) ? m_OutputPaddingSize : 0;
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_BatchedOutputBufferPool.get(), BatchResponseComplete, &request_block->m_BatchResponseMetadata),
        "Setting response callback");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");
    m_CurrentRequest = nullptr;
    m_QueuedSamples.clear();
}

void Server_SUT::IssueContiguousBatchDLRMQuery(const std::vector<mlperf::QuerySample>& samples,
    size_t batch_start_sample, size_t batch_end_sample, size_t num_user_item_pairs,
    const std::vector<size_t>& batchNumPairs)
{
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE

    auto request_block = RequestPool::Obtain(0);

    request_block->m_BatchResponseMetadata.m_TracePtr = trace;
    request_block->m_BatchResponseMetadata.m_DlrmNumPairsList = batchNumPairs;
    request_block->m_BatchResponseMetadata.m_ResponseId.clear();
    request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.clear();
    request_block->m_BatchResponseMetadata.m_RequestBatchSize = m_MaxBatchSize;

    for(size_t i = batch_start_sample; i < batch_end_sample; i++)
    {
        request_block->m_BatchResponseMetadata.m_ResponseId.emplace_back(samples[i].id);
        request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.emplace_back(samples[i].index);
    }
    // Inputs will need to be re-added as the shape is different from run
    // to run
    TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

    auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
    // new batch size for the request
    auto l_InputTensors = m_InputTensors;

    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*) qsl->GetSampleAddress(samples[batch_start_sample].index,
            idx); // Get address of the query
        const size_t single_sample_size = qsl->GetSampleSize(idx);

        auto& shape = std::get<2>(l_InputTensors[idx]);
        shape[0] = num_user_item_pairs;
        if (num_user_item_pairs % 2)
        {
            shape[0] += 1;
            FAIL_IF_ERR(
                TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                "re-adding input");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                            single_sample_size * num_user_item_pairs, m_InputMemoryType, 0),
                "appending input data");
            // Add padding buffer
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                            input_data, single_sample_size, m_InputMemoryType, 0),
                        "appending input data padding");

            if(m_StartFromDevice && m_NumGPUs > 1)
            {
                for(int i = 0; i < m_NumGPUs; i++)
                {
                    input_data =
                        (int8_t*)qsl->GetSampleAddress(samples[batch_start_sample].index, idx, 0,
                                                       i); // Get address of the query for device

                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size * num_user_item_pairs,
                                    m_InputMemoryType, i, m_HostPolicyNames[i].c_str()),
                                "appending input data");
                    // Add padding buffer
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size, m_InputMemoryType, i,
                                    m_HostPolicyNames[i].c_str()),
                                "appending input data padding");
                }
            }
        }
        else
        {
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                            std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                        "re-adding input");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                            input_data, single_sample_size * num_user_item_pairs, m_InputMemoryType,
                            0),
                        "appending input data");

            if(m_StartFromDevice && m_NumGPUs > 1)
            {
                for(int i = 0; i < m_NumGPUs; i++)
                {
                    input_data =
                        (int8_t*)qsl->GetSampleAddress(samples[batch_start_sample].index, idx, 0,
                                                       i); // Get address of the query for device

                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, single_sample_size * num_user_item_pairs,
                                    m_InputMemoryType, i, m_HostPolicyNames[i].c_str()),
                                "appending input data");
                }
            }
        }
    }

    // If there are more than one QSL, we want to add input buffer with device affinity,
    // use the proper host policy name recognized by Triton.
    if (m_SampleLibraries.size() > 1)
    {
        for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
        {
            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[qsl_idx].get());
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*) qsl->GetSampleAddress(samples[batch_start_sample].index,
                    idx); // Get address of the query
                const size_t single_sample_size = qsl->GetSampleSize(idx);
                if (num_user_item_pairs % 2)
                {
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                    single_sample_size * num_user_item_pairs, m_InputMemoryType, 0,
                                    m_HostPolicyNames[qsl_idx].c_str()),
                        "appending input data");
                    // Add padding buffer
                    FAIL_IF_ERR(
                        TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size, m_InputMemoryType, 0,
                            m_HostPolicyNames[qsl_idx].c_str()),
                        "appending input data padding");
                }
                else
                {
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                    single_sample_size * num_user_item_pairs, m_InputMemoryType, 0,
                                    m_HostPolicyNames[qsl_idx].c_str()),
                        "appending input data");
                }
            }
        }
    }
    request_block->m_ResponseMetadata.m_PaddingSize = (num_user_item_pairs % 2) ? m_OutputPaddingSize : 0;

    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_BatchedOutputBufferPool.get(), BatchResponseComplete, &request_block->m_BatchResponseMetadata),
        "Setting response callback");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

void Server_SUT::IssueTritonContiguousBatch(
    const std::vector<mlperf::QuerySample>& samples, size_t start_idx, size_t end_idx)
{
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE

    auto request_block = RequestPool::Obtain(0);

    request_block->m_BatchResponseMetadata.m_TracePtr = trace;
    request_block->m_BatchResponseMetadata.m_ResponseId.clear();
    request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.clear();
    request_block->m_BatchResponseMetadata.m_RequestBatchSize = m_MaxBatchSize;

    for(size_t i = start_idx; i < end_idx; i++)
    {
        request_block->m_BatchResponseMetadata.m_ResponseId.emplace_back(samples[i].id);
        request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.emplace_back(samples[i].index);
    }

    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*)m_SampleLibraries[0]->GetSampleAddress(
            samples[start_idx].index, idx); // Get address of the query
        size_t input_size = m_SampleLibraries[0]->GetSampleSize(idx) * (end_idx - start_idx);

        FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                        request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
                    "removing input data");

        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                        request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                        input_size, m_InputMemoryType, 0),
                    "appending input data");

        // If starting query from device and there are multiple GPUs, provide copies for each GPU
        if(m_StartFromDevice && m_NumGPUs > 1)
        {
            for(int i = 0; i < m_NumGPUs; i++)
            {
                input_data = (int8_t*)m_SampleLibraries[0]->GetSampleAddress(
                    samples[start_idx].index, idx, i); // Get address of the query for device

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                input_data, input_size, m_InputMemoryType, i,
                                m_HostPolicyNames[i].c_str()),
                            "appending input data with host policy");
            }
        }
    }
    // If there are more than one QSL, we want to add input buffer with device affinity,
    // use the proper host policy name recognized by Triton.
    if (m_SampleLibraries.size() > 1)
    {
        for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
        {
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data
                    = (int8_t*) m_SampleLibraries[qsl_idx]->GetSampleAddress(samples[start_idx].index, idx); // Get address of the query
                size_t input_size = m_SampleLibraries[qsl_idx]->GetSampleSize(idx) * (end_idx - start_idx);

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0,
                                m_HostPolicyNames[qsl_idx].c_str()),
                    "appending input data");
            }
        }
    }
    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_BatchedOutputBufferPool.get(), BatchResponseComplete, &request_block->m_BatchResponseMetadata),
        "appending input data");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

void Server_SUT::IssueQueryInternal(const std::vector<mlperf::QuerySample>& samples, size_t start_idx, size_t end_idx)
{
    // Currently BERT is the only model where dynamic
    bool isBertBenchmark = m_IsDynamic;
    bool isSingleQuery = samples.size() == 1;
    bool isDLRMBenchmark = m_UseDlrmQsl;

    // Avoid allocations in single-stream code-path, handle single query and
    // return
    if (isSingleQuery && !isDLRMBenchmark)
    {
        if (isBertBenchmark)
        {
            HandleSingleBertQuery(samples, 0, 0);
        }
        else
        {
            HandleSingleQuery(samples, 0, 0);
        }
        return;
    }

    size_t pool_idx = start_idx == 0 ? 0 : 1;

    if (isBertBenchmark)
    {
        std::vector<std::pair<int, int>> sequenceSamplePosAndLength(samples.size());
        for (int samplePos = 0; samplePos < samples.size(); ++samplePos)
        {
            sequenceSamplePosAndLength[samplePos] = std::make_pair(
                samplePos, static_cast<int>(GetSampleLength(m_SampleLibraries[0], samples[samplePos].index)));
        }
        // Sort the samples according to sequence length
        // Sort samples in the descending order of sentence length
        std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool { return a.second > b.second; });
        for (size_t i = start_idx; i < end_idx; i++)
        {
            // If its bert used samples in the sorted order
            int indexIntoQuerySample = isBertBenchmark ? sequenceSamplePosAndLength[i].first : i;
            HandleSingleBertQuery(samples, indexIntoQuerySample, pool_idx);
        }
    }
    else
    {
        for (size_t i = start_idx; i < end_idx; i++)
        {
            if (isDLRMBenchmark)
            {
                HandleSingleDlrmQuery(samples, i, pool_idx);
            }
            else
            {
                HandleSingleQuery(samples, i, pool_idx);
            }
        }
    }
}

void Server_SUT::Completion(TRITONSERVER_InferenceResponse* response, const ResponseMetaData* response_metadata)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Extract and process the response data */
    const char* name;
    TRITONSERVER_DataType datatype;
    void* userp;
    const int64_t* shape;
    uint64_t dim_count;
    const void* output0_content;
    size_t output0_byte_size;
    TRITONSERVER_MemoryType output0_memory_type;
    int64_t output0_memory_type_id;

    FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(response, 0 /* index */, &name, &datatype, &shape, &dim_count,
                    &output0_content, &output0_byte_size, &output0_memory_type, &output0_memory_type_id, &userp),
        "getting output0 result");
    // Recast the output pointer as a uintptr_t (for LoadGen)
    uintptr_t output0_result = reinterpret_cast<uintptr_t>(output0_content);

    /* Call QuerySamplesComplete */
    mlperf::QuerySampleResponse loadgen_response{
        response_metadata->m_ResponseId, output0_result, output0_byte_size - response_metadata->m_PaddingSize};

    // callback if it exists
    if (m_ResponseCallback)
    {
        std::vector<::mlperf::QuerySampleIndex> response_indices = {response_metadata->m_QuerySampleIdx};
        m_ResponseCallback(&loadgen_response, response_indices, 1);
    }

    QuerySamplesComplete(&loadgen_response, 1, output0_memory_type_id);
}

void Server_SUT::BatchCompletion(TRITONSERVER_InferenceResponse* response,
                                 const BatchResponseMetaData* response_metadata)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Extract and process the response data */
    const char* name;
    TRITONSERVER_DataType datatype;
    void* userp;
    const int64_t* shape;
    uint64_t dim_count;
    const void* output0_content;
    size_t output0_byte_size;
    TRITONSERVER_MemoryType output0_memory_type;
    int64_t output0_memory_type_id;
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(response, 0 /* index */, &name, &datatype, &shape, &dim_count,
                    &output0_content, &output0_byte_size, &output0_memory_type, &output0_memory_type_id, &userp),
        "getting output0 result");

    // Recast the output pointer as a uintptr_t (for LoadGen)
    uintptr_t output0_result = reinterpret_cast<uintptr_t>(output0_content);

    // Construct response list from Inference response
    std::vector<mlperf::QuerySampleResponse> loadgen_responses;

    if (!m_UseDlrmQsl)
    {
        size_t batch1_output_size = output0_byte_size / m_MaxBatchSize;

        auto buffer_ptr = static_cast<const int8_t*>(output0_content);

        for (int i = 0; i < (response_metadata->m_ResponseId).size(); i++)
        {
            loadgen_responses.emplace_back(
                mlperf::QuerySampleResponse{(response_metadata->m_ResponseId)[i], output0_result, batch1_output_size});
            const void* buffer_ptr_inc = buffer_ptr + (batch1_output_size * (i + 1));
            output0_result = reinterpret_cast<uintptr_t>(buffer_ptr_inc);
        }
    }
    else
    {
        int numQueryResponses = response_metadata->m_ResponseId.size();
        auto buffer_ptr = static_cast<const int8_t*>(output0_content);
        int cumulativeNumPairs = 0;

        for (int i = 0; i < numQueryResponses; i++)
        {
            loadgen_responses.emplace_back(mlperf::QuerySampleResponse{(response_metadata->m_ResponseId)[i],
                output0_result, response_metadata->m_DlrmNumPairsList[i] * TRITONSERVER_DataTypeByteSize(datatype)});
            cumulativeNumPairs += response_metadata->m_DlrmNumPairsList[i];
            const void* buffer_ptr_inc = buffer_ptr + cumulativeNumPairs * TRITONSERVER_DataTypeByteSize(datatype);
            output0_result = reinterpret_cast<uintptr_t>(buffer_ptr_inc);
        }
    }
    // callback if it exists
    if (m_ResponseCallback)
    {
        std::vector<::mlperf::QuerySampleIndex> sample_indices = response_metadata->m_QuerySampleIdxList;
        m_ResponseCallback(&loadgen_responses[0], sample_indices,
                           (response_metadata->m_QuerySampleIdxList).size());
    }

    QuerySamplesComplete(&loadgen_responses[0], response_metadata->m_ResponseId.size(),
                         output0_memory_type_id);
}

void Server_SUT::Done()
{
    RequestPool::Destroy();

    /* Delete the response allocator since we are done with it */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(m_Allocator), "deleting response allocator");

    /* Reset the server pointer to nullptr to ensure Init() is called before the
     * server is used
     * again */
    m_Server = nullptr;
}

void Server_SUT::FlushQueries()
{
    TRITONSERVER_InferenceTrace* trace = nullptr;

#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Flush Request START");
    }
#endif // TRITON_FRONTEND_TRACE
    if (m_BatchTritonRequests && m_CurrentRequest != nullptr)
    {
        IssueCurrentBatchedRequest(m_CurrentRequest, trace);
    }
}

void Server_SUT::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) {}

}; // namespace triton_frontend
