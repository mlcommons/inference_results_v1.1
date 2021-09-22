"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import multiprocessing
import threading
import time

import mlperf_loadgen as lg
import numpy as np
from shutil import copyfile

from items import Item
from items import OItem

# add dlrm code path
try:
    dlrm_dir_path = os.environ['DLRM_DIR']
    sys.path.append(dlrm_dir_path)
except KeyError:
    print("ERROR: Please set DLRM_DIR environment variable to the dlrm code location")
    sys.exit(0)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

num_sockets = int(os.getenv('NUM_SOCKETS', 8))
cpus_per_socket = int(os.getenv('CPUS_PER_SOCKET', 28))
cpus_per_process = int(os.getenv('CPUS_PER_PROCESS', 28))
procs_per_socket = cpus_per_socket // cpus_per_process
total_procs = num_sockets * procs_per_socket
cpus_per_instance = int(os.getenv('CPUS_PER_INSTANCE', 14))

NANO_SEC = 1e9

# the datasets we support
DATASETS_KEYS = ["kaggle", "terabyte"]

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-kaggle-pytorch": {
        "dataset": "kaggle",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 128,
    },
    "dlrm-terabyte-pytorch": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

start_time = 0
item_good = 0
item_total = 0
total_instances = 0
item_timing = []
item_results = []
last_timeing = []


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the mlperf model, ie. dlrm")
    parser.add_argument("--model-path", required=True, help="path to the model file")
    parser.add_argument("--dataset", choices=DATASETS_KEYS, help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--enable-profiling", type=bool, default=False, help="enable pytorch profiling")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    parser.add_argument("--test-num-workers", type=int, default=0, help='# of workers reading the data')
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)
    parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs (currently not used)")
    parser.add_argument("--outputs", help="model outputs (currently not used)")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--use-ipex", action="store_true", default=False)
    parser.add_argument("--use-bf16", action="store_true", default=False)
    parser.add_argument("--use-int8", action="store_true", default=False)
    parser.add_argument('--int8-calibration', action='store_true', default=False, help='enable ipex mix precision')
    parser.add_argument('--int8-configuration-dir', default='int8_configure.json', type=str, metavar='PATH',
                            help = 'path to int8 configures, default file name is configure.json')
    parser.add_argument("--threads", default=1, type=int, help="threads")
    parser.add_argument("--cache", type=int, default=0, help="use cache (currently not used)")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--config", default="../mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-config", default="./user.conf", help="mlperf rules user config")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--duration", type=int, help="duration in milliseconds (ms)")
    parser.add_argument("--target-qps", type=int, help="target/expected qps")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--count-samples", type=int, help="dataset items to use")
    parser.add_argument("--count-queries", type=int, help="number of queries to use")
    parser.add_argument("--samples-per-query-multistream", type=int, help="query length for multi-stream scenario (in terms of aggregated samples)")
    # --samples-per-query-offline is equivalent to perf_sample_count
    parser.add_argument("--samples-per-query-offline", type=int, default=2048, help="query length for offline scenario (in terms of aggregated samples)")
    parser.add_argument("--samples-to-aggregate-fix", type=int, help="number of samples to be treated as one")
    parser.add_argument("--samples-to-aggregate-min", type=int, help="min number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-max", type=int, help="max number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-quantile-file", type=str, help="distribution quantile used to generate number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-trace-file", type=str, default="dlrm_trace_of_aggregated_samples.txt")
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.numpy_rand_seed)

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args

def get_dataset(args):
    import criteo

    kwargs = {"randomize": 'total',  "memory_map": True}
    # --count-samples can be used to limit the number of samples used for testing
    ds = criteo.Criteo(data_path=args.dataset_path,
                        name=args.dataset,
                        pre_process=criteo.pre_process_criteo_dlrm,  # currently an identity function
                        use_cache=args.cache,  # currently not used
                        count=args.count_samples,
                        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
                        samples_to_aggregate_min=args.samples_to_aggregate_min,
                        samples_to_aggregate_max=args.samples_to_aggregate_max,
                        samples_to_aggregate_quantile_file=args.samples_to_aggregate_quantile_file,
                        samples_to_aggregate_trace_file=args.samples_to_aggregate_trace_file,
                        test_num_workers=0,
                        max_ind_range=args.max_ind_range,
                        sub_sample_rate=args.data_sub_sample_rate,
                        mlperf_bin_loader=True,
                        **kwargs)
    return ds

def get_device(use_gpu, use_ipex):
    import intel_pytorch_extension as ipex
    if use_ipex:
        device = ipex.DEVICE
    elif use_gpu:
        device = "cuda:0"
    else:
        device = "cpu"
    return device


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, ds_queue, lock, init_counter, proc_num, args, first_instance_start_core):
        multiprocessing.Process.__init__(self)
        global cpus_per_socket
        global cpus_per_process
        global procs_per_socket
        global get_device
        import torch
        import intel_pytorch_extension as ipex
        self.args = args
        self.device = get_device(args.use_gpu, args.use_ipex)
        self.ipex_conf = None
        if args.use_int8:
            self.ipex_conf = ipex.AmpConf(torch.int8, args.int8_configuration_dir)
        self.lock = lock
        self.proc_num = proc_num
        self.ds_queue = ds_queue
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.init_counter = init_counter

        self.workers = []
        self.instances_start_core = []
        self.instances_end_core = []
        #self.instances_affinity = []
        self.instances_core_nums = []
        self.instances_core_id = []
        socket_num = self.proc_num // procs_per_socket
        socket_proc_idx = self.proc_num % procs_per_socket
        self.start_core_idx = socket_num * cpus_per_socket + socket_proc_idx * cpus_per_process
        self.end_core_idx = self.start_core_idx + cpus_per_process
        if self.proc_num == 0:
            self.start_core_idx = self.start_core_idx + first_instance_start_core  #reserved threads for loadgen staff
        self.affinity = range(self.start_core_idx, self.end_core_idx)
        self.core_nums = self.end_core_idx - self.start_core_idx
        self.num_ins = self.core_nums // cpus_per_instance
        if self.proc_num == 0 and (first_instance_start_core > 0):
            left_cores = self.core_nums - self.num_ins * cpus_per_instance
            if left_cores >= (cpus_per_instance // 2):
                self.num_ins = self.num_ins + 1

        for i in range(self.num_ins):
            if self.proc_num == 0:
              if (cpus_per_instance - first_instance_start_core) >= (cpus_per_instance // 2):
                  if i == 0:
                      self.instances_start_core.append(first_instance_start_core)
                      self.instances_core_id.append(first_instance_start_core)
                      self.instances_end_core.append(cpus_per_instance)
                  else:
                      self.instances_start_core.append(i * cpus_per_instance)
                      self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance)
                      self.instances_core_id.append(i * cpus_per_instance)
              else:
                  self.instances_start_core.append((i + 1) * cpus_per_instance)
                  self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance)
                  self.instances_core_id.append((i + 1) * cpus_per_instance)
            else:
                  self.instances_start_core.append(self.start_core_idx + i * cpus_per_instance)
                  self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance)
                  self.instances_core_id.append(i * cpus_per_instance)
            #self.instances_affinity.append(range(self.instances_start_core[i], self.instances_end_core[i]))
            self.instances_core_nums.append(self.instances_end_core[i] - self.instances_start_core[i])

    def input_wrap(self, X, lS_o, lS_i):
        if self.args.use_gpu or self.args.use_ipex:
           lS_i = [S_i.to(self.device) for S_i in lS_i] if isinstance(lS_i, list) else lS_i.to(self.device)
           lS_o = [S_o.to(self.device) for S_o in lS_o] if isinstance(lS_o, list) else lS_o.to(self.device)
           X = X.to(self.device)
           return X, lS_o, lS_i

    def model_predict(self, batch_dense_X, batch_lS_o, batch_lS_i):
        import torch
        import intel_pytorch_extension as ipex
        X, lS_o, lS_i = self.input_wrap(batch_dense_X, batch_lS_o, batch_lS_i)
        if self.args.use_int8:
            with torch.no_grad():
                with ipex.AutoMixPrecision(self.ipex_conf):
                    output = self.model(X, lS_o, lS_i)
        else:
            with torch.no_grad():
                output = self.model(X, lS_o, lS_i)
        return output

    def warmup(self):
        import torch
        for s in range(self.args.max_batchsize, self.args.max_batchsize + 800, 100):
            batch_dense_X = torch.tensor(np.random.normal(-1.0, 1.0, size=(s, 13)).astype(np.float32))
            batch_lS_i = torch.ones([26, s], dtype=torch.long)
            batch_lS_o = torch.LongTensor(26, s)
            for i in range(26):
                batch_lS_o[i] = torch.arange(s)
            self.model_predict(batch_dense_X, batch_lS_o, batch_lS_i)

    def get_samples(self, id_list):
        import torch
        import intel_pytorch_extension as ipex
        ls = []
        num = 0
        for i in id_list:
            ls.append(self.items_in_memory[i])
            num = num + 1

        ls_t = list(zip(*ls))
        X = ipex.core.concat_all_continue(ls_t[0], 0)
        # X = torch.cat(ls_t[0])
        lS_i = ipex.core.concat_all_continue(ls_t[2], 1)
        # lS_i = torch.cat(ls_t[2], dim=1)
        (num_s, len_ls) = lS_i.size()
        lS_o = torch.LongTensor(num_s, len_ls)
        for i in range(num_s):
            lS_o[i] = torch.arange(len_ls)
        #lS_o = torch.stack([torch.arange(len_ls) for _ in range(num_s)])
        T = ipex.core.concat_all_continue(ls_t[3], 0)
        # T = torch.cat(ls_t[3])
        return (X, lS_o, lS_i, T)

    def handle_tasks(self, i, ds_queue, task_queue, result_queue, args, pid):
        import torch
        import intel_pytorch_extension as ipex

        global cpus_per_process
        torch.set_grad_enabled(False)
        ipex.core.enable_auto_dnnl()
        ipex.core.set_execution_mode(False)
        #os.sched_setaffinity(self.workers[i].pid, self.instances_affinity[i])
        ipex.core.thread_bind(self.proc_num, cpus_per_process, self.instances_core_id[i], self.instances_core_nums[i])
        instance_name = str(pid) + "-" + str(i)
        #print(instance_name, " : Start handle_tasks")
        if args.enable_profiling:
            filename = "dlrm_mlperf_offline_run_" + instance_name + ".prof"

        ds = get_dataset(args)
        sample_list = ds_queue.get()
        ds.load_query_samples(sample_list)
        self.items_in_memory = ds.get_items_in_memory()
        print(instance_name, " : Complete load query samples !!")

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        with torch.autograd.profiler.profile(args.enable_profiling, args.use_gpu) as prof:
            while True:
                qitem = task_queue.get()
                if qitem is None:
                    #print(instance_name, " : Exit")
                    break

                presults = []
                try:
                    batch_dense_X, batch_lS_o, batch_lS_i, batch_T = self.get_samples(qitem.content_id)

                  #with torch.autograd.profiler.profile() as prof:
                    results = self.model_predict(batch_dense_X, batch_lS_o, batch_lS_i)
                    # post_process
                    results = results.detach().cpu().contiguous()
                    presults = ipex.core.concat_all_continue((results, batch_T), 1)
                    # presults = torch.cat((results, batch_T), dim=1)

                    if args.accuracy:
                        good = (results.round() == batch_T).nonzero(as_tuple=False).size(0)
                        total = len(results)
                        result_timing = time.time() - qitem.start

                  #with open(filename, "w") as prof_f:
                  #  prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
                except Exception as ex:  # pylint: disable=broad-except
                    log.error("instance ", instance_name, " failed, %s", ex)
                    presults = [[]] * len(qitem.query_id)
                finally:
                    response_array_refs = []
                    query_list = qitem.query_id
                    idx_offsets = qitem.idx_offsets
                    prev_off = 0
                    for idx, query_id in enumerate(query_list):
                        cur_off = prev_off + idx_offsets[idx]
                        response_array = array.array("B", np.array(presults[prev_off:cur_off], np.float32).tobytes())
                        response_array_refs.append(response_array)
                        prev_off = cur_off
                    if args.accuracy:
                        result_queue.put(OItem(np.array(presults, np.float32), query_list, response_array_refs, good, total, result_timing))
                    else:
                        result_queue.put(OItem([], query_list, response_array_refs))

        if args.enable_profiling:
            with open(filename, "w") as prof_f:
                prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))

        ds.unload_query_samples(sample_list)

    def run(self):

        os.sched_setaffinity(self.pid, self.affinity)
        #cmd = "taskset -p -c %d-%d %d" % (self.start_core_idx, self.end_core_idx, self.pid)
        #os.system(cmd)
        import torch
        import torch.multiprocessing as mp
        from backend_pytorch_native import get_backend

        torch.set_num_threads(self.core_nums)
        # model initialize
        backend = get_backend(self.args.backend, self.args.dataset, self.device, self.args.max_ind_range,
                              self.args.data_sub_sample_rate, self.args.use_gpu, self.args.use_ipex, self.args.use_bf16)
        self.model = backend.load(self.args.model_path, self.args.inputs, self.args.outputs)

        print ('Start warmup.')
        self.warmup()
        print ('Warmup done.')

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        if self.num_ins > 1:
            for i in range(self.num_ins):
                worker = mp.Process(target=self.handle_tasks, args=(i, self.ds_queue, self.task_queue, self.result_queue, self.args, self.pid))
                self.workers.append(worker)

            for w in self.workers:
                w.start()
            for w in self.workers:
                w.join()
        else:
            self.handle_tasks(0, self.ds_queue, self.task_queue, self.result_queue, self.args, self.pid)


class QueueRunner:
    def __init__(self, inQueue, dataset, max_batchsize=128):
        self.inQueue = inQueue
        self.max_batchsize = max_batchsize
        self.ds = dataset
        self.sample_len_dict = {}
        self.sample_len_num = {}
        self.temp_query_buffer = []
        self.temp_idx_buffer = []
        self.temp_idx_offset_buffer = []

    def fill_buffers(self, query_id, idx, sample_len):
        self.temp_query_buffer.append(query_id)
        self.temp_idx_buffer.append(idx)
        self.temp_idx_offset_buffer.append(sample_len)

    def issue(self, query_id, idx, idxlen):
        self.inQueue.put(Item(query_id, idx, idxlen))
        self.temp_query_buffer = []
        self.temp_idx_buffer = []
        self.temp_idx_offset_buffer = []

    def enqueue(self, query_id, idx):
        query_len = len(query_id)
        handled_query = 0
        #issued_tasks = 0
        for i in range(0, query_len):
            sample_len = self.ds.get_sample_length(idx[i])
            cur_sample_len = self.sample_len_num[sample_len] + sample_len
            if  cur_sample_len >= self.max_batchsize:
                while self.sample_len_dict[sample_len]:
                    (temp_query_id, temp_idx_id) = self.sample_len_dict[sample_len].pop()
                    self.fill_buffers(temp_query_id, temp_idx_id, sample_len)
                    handled_query = handled_query + 1
                self.fill_buffers(query_id[i], idx[i], sample_len)
                handled_query = handled_query + 1
                self.issue(self.temp_query_buffer, self.temp_idx_buffer, self.temp_idx_offset_buffer)
                self.sample_len_num[sample_len] = 0
            else:
                self.sample_len_dict[sample_len].append((query_id[i], idx[i]))
                self.sample_len_num[sample_len] = cur_sample_len

        current_bs = 0
        while handled_query < query_len:
            for sample_len in self.sorted_keys:
                while self.sample_len_dict[sample_len]:
                    current_bs = current_bs + sample_len
                    if current_bs > self.max_batchsize:
                        break
                    (temp_query_id, temp_idx_id) = self.sample_len_dict[sample_len].pop()
                    self.fill_buffers(temp_query_id, temp_idx_id, sample_len)
                    handled_query = handled_query + 1 
                if current_bs > self.max_batchsize:
                    break
            self.issue(self.temp_query_buffer, self.temp_idx_buffer, self.temp_idx_offset_buffer)
            current_bs = 0

        # put in end marks
        for _ in range(total_instances):
            self.inQueue.put(None)
        #print("Complete Enqueue, issued tasks is ", issued_tasks)

    def load_query_samples(self, sample_list):
        self.ds.load_query_samples(sample_list)
        self.sample_lengths_list = self.ds.get_sample_lengths_list()
        self.sorted_keys = sorted(self.sample_lengths_list, reverse = True)
        for sample_len in self.sorted_keys:
            self.sample_len_num[sample_len] = 0
            self.sample_len_dict[sample_len] = []

    def unload_query_samples(self, sample_list):
        self.ds.unload_query_samples(sample_list)

    def flush_queries(self):
        self.sample_len_dict = {}
        self.sample_len_num = {}

def add_results(final_results, name, result_dict, result_list, took, show_accuracy=False):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": len(result_list),
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "roc_auc" in result_dict:
            result["roc_auc"] = 100. * result_dict["roc_auc"]
            acc_str += ", roc_auc={:.3f}%".format(result["roc_auc"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print("{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
        name, result["qps"], result["mean"], took, acc_str,
        len(result_list), buckets_str))

def response_loadgen(outQueue, accuracy, lock):
    global item_good
    global item_total
    global item_timing
    global item_results

    '''
    max_outqueue_len = 0
    iter_num = 0
    '''
    while True:
        #iter_num += 1
        oitem = outQueue.get()
        if oitem is None:
            break

        response = []
        if accuracy:
            lock.acquire()

        for q_id, arr in zip(oitem.query_ids, oitem.array_ref):
            bi = arr.buffer_info()
            response.append(lg.QuerySampleResponse(q_id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
        '''
        if iter_num % 20 == 0:
            qsize = outQueue.qsize()
            if qsize > max_outqueue_len:
                max_outqueue_len = qsize
        '''

        if accuracy:
            item_good += oitem.good
            item_total += oitem.total
            item_timing.append(oitem.timing)
            item_results.append(oitem.presults)
            lock.release()

    '''
    if max_outqueue_len > 100:
        print ('out queue piled up to {} items, consider increase numOutQ'.format(max_outqueue_len))
    '''

def main():
    global num_sockets
    global cpus_per_socket
    global cpus_per_process
    global cpus_per_instance
    global total_instances
    global start_time
    global item_total
    global last_timeing

    args = get_args()
    log.info(args)
    config = os.path.abspath(args.config)
    user_config = os.path.abspath(args.user_config)

    if not os.path.exists(config):
        log.error("{} not found".format(config))
        sys.exit(1)

    if not os.path.exists(user_config):
        log.error("{} not found".format(user_config))
        sys.exit(1)

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists("./audit.config"):
            copyfile("./audit.config", output_dir + "/audit.config")
        if args.use_int8 and os.path.exists(args.int8_configuration_dir):
            copyfile(args.int8_configuration_dir, output_dir + "/" + args.int8_configuration_dir)
        os.chdir(output_dir)

    import torch
    import criteo

    cpus_for_loadgen = 1
    left_cores = cpus_per_socket * num_sockets - total_procs * cpus_per_process
    first_instance_start_core = cpus_for_loadgen
    if left_cores > cpus_for_loadgen and num_sockets > cpus_for_loadgen:
        first_instance_start_core = 0
        cpus_for_loadgen = num_sockets

    total_instances = 0
    instances_per_proc = (cpus_per_process // cpus_per_instance)
    for i in range(total_procs):
        if i == 0 and first_instance_start_core > 0:
            first_instances = ((cpus_per_process - first_instance_start_core) // cpus_per_instance)
            total_instances = total_instances + first_instances
            left_cores = cpus_per_process - first_instances * cpus_per_instance
            if (left_cores - first_instance_start_core) >= (cpus_per_instance // 2):
                total_instances = total_instances + 1
        else:
            total_instances = total_instances + instances_per_proc 
    print("Setup {} Instances !!".format(total_instances))

    lock = multiprocessing.Lock()
    init_counter = multiprocessing.Value("i", 0)
    total_samples = multiprocessing.Value("i", 0)
    dsQueue = multiprocessing.Queue()
    numOutQ = total_procs
    outQueues = [multiprocessing.Queue() for i in range(numOutQ)]
    #inQueue = multiprocessing.JoinableQueue()
    inQueue = multiprocessing.Queue()
    consumers = [Consumer(inQueue, outQueues[i%numOutQ], dsQueue, lock, init_counter, i, args, first_instance_start_core) for i in range(total_procs)]
    for c in consumers:
        c.start()

    # Wait until subprocess ready
    while init_counter.value < total_procs: time.sleep(2)

    torch.set_num_threads(cpus_per_socket * num_sockets)

    dlrm_dataset = get_dataset(args)
    total_samples.value = dlrm_dataset.get_item_count()
    scenario = SCENARIO_MAP[args.scenario]
    runner_map = {
        lg.TestScenario.Server: QueueRunner,
        lg.TestScenario.Offline: QueueRunner
    }
    runner = runner_map[scenario](inQueue, dlrm_dataset, max_batchsize=args.max_batchsize)

    # Start response thread
    response_workers = [threading.Thread(
        target=response_loadgen, args=(outQueues[i], args.accuracy, lock)) for i in range(numOutQ)]
    for response_worker in response_workers:
       response_worker.daemon = True
       response_worker.start()

    settings = lg.TestSettings()
    settings.FromConfig(config, args.model, args.scenario)
    settings.FromConfig(user_config, args.model, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        settings.performance_sample_count_override = total_samples.value

    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.duration:
        settings.min_duration_ms = args.duration
        settings.max_duration_ms = args.duration

    if args.target_qps:
        settings.server_target_qps = float(args.target_qps)
        settings.offline_expected_qps = float(args.target_qps)

    if args.count_queries:
        settings.min_query_count = args.count_queries
        settings.max_query_count = args.count_queries

    if args.samples_per_query_multistream:
        settings.multi_stream_samples_per_query = args.samples_per_query_multistream

    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_target_latency_ns = int(args.max_latency * NANO_SEC)

    def load_query_samples(sample_list):
        # Wait until subprocess ready
        global start_time
        runner.load_query_samples(sample_list)
        for _ in range(total_instances):
            dsQueue.put(sample_list)
        while init_counter.value < total_procs + total_instances: time.sleep(2)
        start_time = time.time()

    def unload_query_samples(sample_list):
        runner.unload_query_samples(sample_list)

    def issue_queries(response_ids, query_sample_indexes):
        runner.enqueue(response_ids, query_sample_indexes)

    def flush_queries():
        runner.flush_queries()

    def process_latencies(latencies_ns):
        # called by loadgen to show us the recorded latencies
        global last_timeing
        last_timeing = [t / NANO_SEC for t in latencies_ns]

    sut = lg.ConstructFastSUT(issue_queries, flush_queries, process_latencies)
    sample_num = min(total_samples.value, args.samples_per_query_offline)
    qsl = lg.ConstructQSL(total_samples.value, sample_num, load_query_samples, unload_query_samples)

    log.info("starting {}".format(scenario))
    result_dict = {"good": 0, "total": 0, "roc_auc": 0, "scenario": str(scenario)}

    lg.StartTest(sut, qsl, settings)

    if not last_timeing:
        last_timeing = item_timing
    if args.accuracy:
        result_dict["good"] = item_good
        result_dict["total"] = item_total
        result_dict["roc_auc"] = criteo.auc_score(item_results)

    final_results = {
        "runtime": "pytorch-native-dlrm",
        "version": torch.__version__,
        "time": int(time.time()),
        "cmdline": str(args),
    }

    add_results(final_results, "{}".format(scenario),
                result_dict, last_timeing, time.time() - start_time, args.accuracy)

    for c in consumers:
        c.join()

    for i in range(numOutQ):
        outQueues[i].put(None)

    lg.DestroyQSL(qsl)
    lg.DestroyFastSUT(sut)

    # write final results
    if args.output:
        with open("results.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
