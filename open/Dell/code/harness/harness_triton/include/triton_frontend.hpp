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

#ifndef __TRITON_FRONTEND_HPP__
#define __TRITON_FRONTEND_HPP__

// TRITON
#include "src/servers/tracer.h"
#include "triton/core/tritonserver.h"

// QSL
#include "qsl.hpp"

// LoadGen
#include "system_under_test.h"

// General C++
#include <atomic>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <thread>

#include "triton_frontend_server.hpp"

// NUMA utils
#include "pinned_memory_pool.hpp"
#include "utils.hpp"

namespace triton_frontend {

class Server_SUT : public Triton_Server_SUT
{
  public:
    Server_SUT(std::string name, std::string model_repo_path, std::string model_name,
               uint32_t model_version, bool use_dlrm_qsl, bool start_from_device,
               bool end_on_device, bool pinned_input, uint64_t request_pool_count)
        : Triton_Server_SUT(name, model_repo_path, model_name, model_version, use_dlrm_qsl,
                            start_from_device, end_on_device, pinned_input, request_pool_count)
    {
    }
    ~Server_SUT() {}

    void Init(size_t min_sample_size = 1, size_t max_sample_size = 1,
              size_t buffer_manager_thread_count = 0, bool batch_triton_requests = false,
              bool check_contiguity = false, const std::string& numa_config_str = "");
    void Warmup(double duration_sec, double expected_qps);

    void Completion(TRITONSERVER_InferenceResponse* response,
                    const ResponseMetaData* response_metadata);
    void BatchCompletion(TRITONSERVER_InferenceResponse* response,
                         const BatchResponseMetaData* response_metadata);
    void Done();

    // SUT virtual interface
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples);
    virtual void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns);
    virtual void FlushQueries();

  private:
    void IssueQueryInternal(const std::vector<mlperf::QuerySample>& samples, size_t start_idx,
                            size_t end_idx);
    bool CheckContiguity(const std::vector<mlperf::QuerySample>& samples, int start, int end);
    bool CheckDLRMContiguity(const std::vector<mlperf::QuerySample>& samples, int start, int end, int num_pairs);
    void HandleSingleDlrmQuery(const std::vector<mlperf::QuerySample>& samples, int indexIntoQuerySample, int pool_idx);
    void HandleSingleBertQuery(const std::vector<mlperf::QuerySample>& samples, int indexIntoQuerySample, int pool_idx);
    void HandleSingleQuery(const std::vector<mlperf::QuerySample>& samples, int indexIntoQuerySample, int pool_idx);
    void HandleDlrmBatchedQueries(const std::vector<mlperf::QuerySample>& samples);
    void HandleDlrmServerQuery(const std::vector<mlperf::QuerySample>& samples);
    void IssueTritonContiguousBatch(const std::vector<mlperf::QuerySample>& samples, size_t start_idx, size_t end_idx);
    void IssueContiguousBatchDLRMQuery(const std::vector<mlperf::QuerySample>& samples, size_t batch_start_sample,
        size_t batch_end_sample, size_t num_user_item_pairs, const std::vector<size_t>& batchNumPairs);
    void IssueOrQueueCoalescedDLRMQuery(const std::vector<mlperf::QuerySample>& samples,
                                        size_t batch_start_sample, size_t batch_end_sample,
                                        size_t num_user_item_pairs,
                                        const std::vector<size_t>& batchNumPairs);
    void IssueCurrentBatchedRequest(RequestPool::Block* request_block,
                                    TRITONSERVER_InferenceTrace* trace);

    // Keep a pointer to the "current request" when accumulating server queries
    RequestPool::Block* m_CurrentRequest = nullptr;
    std::vector<mlperf::QuerySample> m_QueuedSamples;
};
}; // namespace triton_frontend

#endif
