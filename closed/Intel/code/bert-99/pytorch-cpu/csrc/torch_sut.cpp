#include <ATen/core/grad_mode.h>
#include <c10/util/TypeCast.h>
#include <vector>
#include <list>
#include <chrono>
#include <algorithm>
#include <condition_variable>
#include <type_traits>
#include <loadgen.h>
#include <query_sample.h>

#include "torch_sut.hpp"


BertServerSUT::BertServerSUT(
    const std::string& model_file,
    const std::string& sample_file,
    int inter_parallel,
    int intra_parallel,
    int batch, bool ht
  ) : qsl_(sample_file), model_(model_file), mThreshold_(batch) ,
  nProcsPerInstance_(intra_parallel), nInstances_(inter_parallel), mHt_(ht) {

  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  if (nProcsPerInstance_ * nInstances_ > nMaxProc)
    nInstances_ = nMaxProc / nProcsPerInstance_;

  // Construct instances
  for (int i = 0; i < nInstances_; ++ i)
    mInstances_.emplace_back(&BertServerSUT::thInstance, this, i);
}

//
// TODO: Use hierachy information to allocate place
//
int BertServerSUT::rootProc(int index) {
  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  // XXX : Assumed 2-sockets, HT on !!!
  int part[] = {nMaxProc, nMaxProc*(2 + (int)mHt_)/4};

  auto select = index & 1;
  auto root = part[select] - nProcsPerInstance_ * ((index>>1) + 1);

  // Assert root > 0
  return root;
}

void BertServerSUT::thInstance(int index) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nProcsPerInstance_);
  auto root = rootProc(index);
  auto which = index & 1;

  for (int i = 0; i < nProcsPerInstance_; ++ i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  Queue_t snippet;
  int count = 0;
  // Wait for work
  while (true) {
    // critical section
    {
      std::unique_lock<std::mutex> l(mtx_);
      ctrl_.wait(l, [this] {return mStop_ || !mQueue_.empty();});

      if (mStop_)
        break;

      auto nPeek = std::min(mQueue_.size(), mThreshold_);
      auto it = mQueue_.begin();
      // XXX: pointer chaser, choose better solution
      std::advance(it, nPeek);
      snippet.clear();
      snippet.splice(snippet.begin(), mQueue_, mQueue_.begin(), it);
      if (!mQueue_.empty()) ctrl_.notify_one();
    }

    std::vector<mlperf::QuerySample> samples(snippet.begin(), snippet.end());

    //
    // Model Inference, switch single/multiple batch model
    // According to length
    //

    for (auto sample : samples ) {
      auto stack = qsl_.GetSample(sample.index);
      auto time_begin = std::chrono::system_clock::now();
      auto results = model_.inference_at(which, stack);
      auto time_end = std::chrono::system_clock::now();
      std::chrono::duration<double> diff = time_end - time_begin;
      count++;
      if (count % 100 == 0) {
         printf("####processed %d samples mQueue_.size()=%d \n", count, (int)mQueue_.size());
      }
      auto tList = qsl_.GetTensorListFrom(results);
      auto tStack = at::stack(tList, -1).squeeze(0);
      QuerySamplesComplete(sample, tStack);
    }
  }
}

void BertServerSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const at::Tensor& results) {
  std::vector<mlperf::QuerySampleResponse> responses(samples.size());

  for (size_t i = 0; i < samples.size(); ++ i) {
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(results[i].data_ptr());
    responses[i].size = results[i].nbytes();
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

void BertServerSUT::QuerySamplesComplete(
    const mlperf::QuerySample& sample,
    const at::Tensor& result) {
  mlperf::QuerySampleResponse response;

  response.id = sample.id;
  response.data = reinterpret_cast<uintptr_t>(result.data_ptr());
  response.size = result.nbytes();

  mlperf::QuerySamplesComplete(&response, 1);
}

void BertServerSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  std::unique_lock<std::mutex> l(mtx_);
  for (auto sample : samples) {
    mQueue_.emplace_back(sample);
  }
  ctrl_.notify_one();
}

BertServerSUT::~BertServerSUT() {
  {
    std::unique_lock<std::mutex> l(mtx_);
    mStop_ = true;
    ctrl_.notify_all();
  }

  for (auto& Instance : mInstances_)
    Instance.join();
}


BertOfflineSUT::BertOfflineSUT(
    const std::string& model_file,
    const std::string& sample_file,
    int inter_parallel,
    int intra_parallel,
    int batch, bool ht
  ) : qsl_(sample_file), model_(model_file), mThreshold_(batch) ,
  nProcsPerInstance_(intra_parallel), nInstances_(inter_parallel), mHt_(ht) {

  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  if (nProcsPerInstance_ * nInstances_ > nMaxProc)
    nInstances_ = nMaxProc / nProcsPerInstance_;

  // Construct instances
  for (int i = 0; i < nInstances_; ++ i)
    mInstances_.emplace_back(&BertOfflineSUT::thInstance, this, i);
}

//
// TODO: Use hierachy information to allocate place
//
int BertOfflineSUT::rootProc(int index) {
  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  // XXX : Assumed 2-sockets, HT on !!!
  int part[] = {nMaxProc, nMaxProc*(2 + (int)mHt_)/4};

  auto select = index & 1;
  auto root = part[select] - nProcsPerInstance_ * ((index>>1) + 1);

  // Assert root > 0
  return root;
}

void BertOfflineSUT::thInstance(int index) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nProcsPerInstance_);
  auto root = rootProc(index);
  auto which = index & 1;

  for (int i = 0; i < nProcsPerInstance_; ++ i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  Queue_t snippet;

  // Wait for work
  while (true) {
    // critical section
    {
      std::unique_lock<std::mutex> l(mtx_);
      ctrl_.wait(l, [this] {return mStop_ || !mQueue_.empty();});

      if (mStop_)
        break;

      auto nPeek = std::min(mQueue_.size(), mThreshold_);
      auto it = mQueue_.begin();
      // XXX: pointer chaser, choose better solution
      std::advance(it, nPeek);
      snippet.clear();
      snippet.splice(snippet.begin(), mQueue_, mQueue_.begin(), it);
      // if (!mQueue_.empty()) Offline never empty
      ctrl_.notify_one();
    }

    std::vector<mlperf::QuerySample> samples(snippet.begin(), snippet.end());

    //
    // Model Inference, switch single/multiple batch model
    // According to length
    //
    if (qsl_.GetFeatureLength(samples[0].index) < slength_) {
      std::vector<mlperf::QuerySampleIndex> indices (samples.size());

      std::transform(samples.cbegin(), samples.cend(), indices.begin(),
          [](mlperf::QuerySample sample) {return sample.index;});

      auto stack = qsl_.AssembleSamples(std::move(indices));
      auto results = model_.inference_at(which, stack);
      auto tList = qsl_.GetTensorListFrom(results);
      auto tStack = at::stack(tList, -1);
      QuerySamplesComplete(samples, tStack);
    } else {
      for (auto sample : samples ) {
        auto stack = qsl_.GetSample(sample.index);
        auto results = model_.inference_at(which, stack);
        auto tList = qsl_.GetTensorListFrom(results);
        auto tStack = at::stack(tList, -1).squeeze(0);
        QuerySamplesComplete(sample, tStack);
      }
    }
  }
}

void BertOfflineSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const at::Tensor& results) {
  std::vector<mlperf::QuerySampleResponse> responses(samples.size());

  for (size_t i = 0; i < samples.size(); ++ i) {
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(results[i].data_ptr());
    responses[i].size = results[i].nbytes();
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

void BertOfflineSUT::QuerySamplesComplete(
    const mlperf::QuerySample& sample,
    const at::Tensor& result) {
  mlperf::QuerySampleResponse response;

  response.id = sample.id;
  response.data = reinterpret_cast<uintptr_t>(result.data_ptr());
  response.size = result.nbytes();

  mlperf::QuerySamplesComplete(&response, 1);
}

void BertOfflineSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  std::unique_lock<std::mutex> l(mtx_);
  // Parallel sort samples into a queue
  mQueue_ = qsl_.Sort(samples);
  ctrl_.notify_one();
}

BertOfflineSUT::~BertOfflineSUT() {
  {
    std::unique_lock<std::mutex> l(mtx_);
    mStop_ = true;
    ctrl_.notify_all();
  }

  for (auto& Instance : mInstances_)
    Instance.join();
}

// -------------------------------------------------------------------------------
// Server model SUT
//
BertSUT::BertSUT(
    const std::string& model_file,
    const std::string& sample_file,
    int inter_parallel,
    int intra_parallel
  ) : qsl_(sample_file), model_(model_file)
  , nCoresPerInstance_(intra_parallel), nInstances_(inter_parallel)
  , aUnitsMap_( (2<<nInstances_) -1 ) {

  if (nCoresPerInstance_ * nInstances_ > Instance::getMaxProc())
    nInstances_ = Instance::getMaxProc() / nCoresPerInstance_;

  // Construct instances
  for (int i = 0; i < nInstances_; ++ i)
    mInstances_.emplace_back(i, *this);
}

BertSUT::Instance::Instance(int index, BertSUT &sut)
  : index_(index),
  sut_(sut), mUnits_(0), mStop_(false),
  th_(&BertSUT::Instance::root_thread, this) {
}

// From last to first
int BertSUT::Instance::RootPlace() const {
  return (Instance::getMaxProc() - sut_.nCoresPerInstance_ * (index_+1));
}

// Enque samples and single the Unit
void BertSUT::Instance::inference(const std::vector<mlperf::QuerySample>& samples) {
}

void BertSUT::Instance::root_thread() {
  // Pin root thread to root core
  pinRoot(RootPlace());

  while (true) {
    // std::unique_lock<std::mutex> l(mtx_);
    // ctrl_.wait(l, [this] {return mStop_;});

    if (mStop_)
      break;

    // Re-pin threads if affinity settings changed
    pinThreads();

    // Process the samples

    // Free Compute Units
    sut_.FreeUnits(getUnitsMask());
    // Pack the response
  }
}
void BertSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  IssueQueryOffline(samples);

  // TODO: Server model
  //
  // allocate working units
  // TODO:
  //   1. Chop samples into sample set
  //   2. Allocate compute resource according to each sample set
  //
}

void BertSUT::IssueQueryOffline(const std::vector<mlperf::QuerySample>& samples) {
  // Configure all instance to take one compute unit
  // TODO: Reverse position from last to first
  //
  for (auto &instance : mInstances_) {
    auto unitsMask = AllocUnits(1);

    auto places = MaskToPlaces(CoreMask(unitsMask, nCoresPerInstance_));

    instance.setAffinityPlaces(std::move(places));
    instance.setUnitsMask(unitsMask);
  }
}

// Spinlock? or lock
uint64_t BertSUT::AllocUnits(int nUnits) {
  while (__builtin_popcountl(aUnitsMap_.load()) < nUnits) {
    std::this_thread::yield();
  }

  // We have enough Units
  auto mask = aUnitsMap_.load();
  auto ret = 0;

  for (int i = 0; i < nUnits; ++ i) {
    auto pos = __builtin_ffs(mask);
    ret |= 1<<pos;
    mask &= ~(1<<pos);
  }

  // Assume only loadgen allocates units
  // Instances free units
  aUnitsMap_.fetch_and(~ret);

  return ret;
}

//
// Allocate units in amount of time frame (in ms)
// Return requested unit mask if success
// or
// Return 0
//
uint64_t BertSUT::AllocUnits(int nUnits, uint64_t timeMs) {
  using ms = std::chrono::milliseconds;
  using timer = std::chrono::high_resolution_clock;
  auto duration = ms(timeMs);

  auto start = timer::now();
  while (__builtin_popcountl(aUnitsMap_.load()) < nUnits) {
    std::this_thread::yield();

    if (timer::now() - start >= duration) {
      return 0;
    }
  }

  // We have enough Units
  auto mask = aUnitsMap_.load();
  auto ret = 0;

  for (int i = 0; i < nUnits; ++ i) {
    auto pos = __builtin_ffs(mask);
    ret |= 1<<pos;
    mask &= ~(1<<pos);
  }

  // Assume only loadgen allocates units
  // Instances free units
  aUnitsMap_.fetch_and(~ret);

  return ret;
}

void BertSUT::FreeUnits(uint64_t mask) {
  aUnitsMap_.fetch_or(mask);
}

std::vector<int> BertSUT::MaskToPlaces(uint64_t mask) {
  std::vector<int> ret;

  while (mask != 0) {
    auto pos = __builtin_ffs(mask);
    mask &= ~(1<<pos);
    ret.emplace_back(pos);
  }

  return ret;
}

uint64_t BertSUT::PlacesToMask(std::vector<int> places) {
  uint64_t mask = 0;

  for (auto place : places)
    mask |= 1<<place;

  return mask;
}

uint64_t BertSUT::CoreMask(uint64_t unitsMask, int unitCores) {
  auto coreMasks = (1<<unitCores) - 1;
  uint64_t ret = 0;

  while (unitsMask != 0) {
    auto pos = __builtin_ffs(unitsMask);
    coreMasks &= ~(1<<pos);
    ret |= coreMasks << (pos * unitCores);
  }

  return ret;
}
