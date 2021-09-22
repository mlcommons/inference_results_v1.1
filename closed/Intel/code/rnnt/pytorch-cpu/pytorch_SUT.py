# copyright (c) 2020, Cerebras Systems, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

import array
import numpy as np
import toml
import mlperf_loadgen as lg
from tqdm import tqdm

from QSL import AudioQSL, AudioQSLInMemory
from helpers import add_blank_label

import threading
import time
#import torch
#import torch.autograd.profiler as profiler
#import torch.multiprocessing as mp
import multiprocessing as mp

query_count = 0
finish_count = 0
debug = False
start_time = time.time()
PROFILE_RANK = [0]

def get_num_cores():
    cmd = "lscpu | awk '/^Core\(s\) per socket:/ {cores=$NF}; /^Socket\(s\):/ {sockets=$NF}; END{print cores*sockets; print cores; print sockets}'"
    lscpu = os.popen(cmd).readlines()
    return int(str.rstrip(lscpu[0])), int(str.rstrip(lscpu[1])), int(str.rstrip(lscpu[2]))

def block_until(counter, num_ins, t):
    while counter.value < num_ins:
        time.sleep(t)

def trace_handler(prof, trace_path=None):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=40))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=40))
    if trace_path:
        prof.export_chrome_trace(trace_path)

class Input(object):
    def __init__(self, id_list, idx_list):
        assert isinstance(id_list, list)
        assert isinstance(idx_list, list)
        assert len(id_list) == len(idx_list)
        self.query_id_list = id_list
        self.query_idx_list = idx_list

class EncodeOutput(object):
    def __init__(self, id_list, idx_list, logits, logits_lens, dur_fea, dur_enc):
        assert isinstance(id_list, list)
        assert isinstance(idx_list, list)
        assert len(id_list) == len(idx_list)
        self.query_id_list = id_list
        self.query_idx_list = idx_list
        self.logits = logits
        self.logits_lens = logits_lens
        self.dur_enc = dur_enc
        self.dur_fea = dur_fea

class Output(object):
    def __init__(self, query_id, transcript):
        self.query_id = query_id
        self.transcript = transcript

class AddNone(object):
    def __init__(self):
        self.value = 0


class InQueue():
    def __init__(self, input_queue_list, qsl, seq_cutoff_list,
                 batch_size_list):

        self.input_queue_list = input_queue_list
        self.qsl = qsl
        self.seq_cutoff_list = seq_cutoff_list
        self.num_queues = len(input_queue_list)
        self.batch_size_list = batch_size_list
        self.query_batcher = [[] for _ in range(self.num_queues)]
        # record the first arrival time of the query batch
        self.query_batcher_time = [None for _ in range(self.num_queues)]
        self.curr_query_count = 0

    def put(self, query_samples):

        if query_samples==None:
            # no more calls to put function
            # submit remaining queries in query batcher to input queues
            # process remaining queries with BS=1
            for i in range(self.num_queues):
                for q in self.query_batcher[i]:
                    input_item = Input([q.id], [q.index])
                    self.input_queue_list[i].put(input_item)
            return

        self.curr_query_count += len(query_samples)

        for sample in query_samples:
            for i in range(self.num_queues):
                idx = sample.index  #BS=1
                waveform = self.qsl[idx]
                if len(waveform) <= self.seq_cutoff_list[i]:
                    if self.query_batcher[i] == []:
                        self.query_batcher_time[i] = time.time()
                    self.query_batcher[i].append(sample)
                    # put queries in queue if BS treshold reached
                    if len(self.query_batcher[i]) == self.batch_size_list[i]:
                        qid_list, qidx_list = [], []
                        for q in self.query_batcher[i]:
                          qid_list.append(q.id)
                          qidx_list.append(q.index)
                        input_item = Input(qid_list, qidx_list)
                        self.input_queue_list[i].put(input_item)
                        self.query_batcher[i] = []
                        self.query_batcher_time[i] = None
                    break
        for i in range(self.num_queues):
            if self.query_batcher_time[i] != None and time.time() - self.query_batcher_time[i] > 0.1:
                #print ("issue sample in queue {} because time is pressing, samples in queue {}".format(i, len(self.query_batcher[i])))
                qid_list, qidx_list = [], []
                for q in self.query_batcher[i]:
                  qid_list.append(q.id)
                  qidx_list.append(q.index)
                input_item = Input(qid_list, qidx_list)
                self.input_queue_list[i].put(input_item)
                self.query_batcher[i] = []
                self.query_batcher_time[i] = None

class Consumer(mp.Process):
    def __init__(self, task_queue, task_queue_group, result_queue, lock, init_counter,
                 rank, core_list, qsl, config_toml, checkpoint_path, dataset_dir,
                 manifest_filepath, perf_count, profile, int8, bf16, warmup,
                 configure_path, mode, total_fea=0, total_enc=0, total_dec=0, verbose="0"):

        mp.Process.__init__(self)

        ### sub process
        self.task_queue = task_queue
        self.task_queue_group = task_queue_group
        self.result_queue = result_queue
        if (result_queue != None):
            self.result_queue.put(AddNone())
        self.lock = lock
        self.init_counter = init_counter
        self.rank = rank
        self.core_list = core_list
        self.num_cores = len(self.core_list)

        self.qsl = qsl
        self.config_toml = config_toml
        self.checkpoint_path = checkpoint_path
        self.dataset_dir = dataset_dir
        self.manifest_filepath = manifest_filepath
        self.perf_count = perf_count
        self.profile = profile
        self.bf16 = bf16
        self.int8 = int8
        self.configure_path = configure_path
        self.warmup = warmup
        self.verbose = verbose
        self.queue_wait = 0.0
        self.queue_count = 0
        # by default, when get a none, the input queue is empty
        # however we can add 'none count' by add input with non-count
        self.wait_for_none_count = 1

        self.model_init = False
        self.mode = mode
        self.total_fea = total_fea
        self.total_enc = total_enc
        self.total_dec = total_dec
        self.consumer_fea = 0
        self.consumer_enc = 0
        self.consumer_dec = 0
        self.prof_fea_list = []
        self.prof_enc_list = []
        self.prof_dec_list = []

    # warmup basically go through samples with different feature lengths so
    # all shapes can be prepared
    def do_warmup(self):
        if not self.warmup:
            return
        if self.rank in PROFILE_RANK:
            print('Start Warmup')
        import torch
        import intel_pytorch_extension as ipex
        conf = ipex.AmpConf(torch.int8, self.configure_path) if self.int8 else None
        t0 = time.time()
        max_len = 500
        batch_size = 384
        features, features_lens = self.generate_data(max_len, batch_size)
        logits, logits_lens, _ = self.predict_enc(features, features_lens, conf)
        _, _ = self.predict_dec(logits, logits_lens, batch_size)

        t1 = time.time()
        if self.rank in PROFILE_RANK:
            print('Warmup done, cost {:.3f}s'.format(t1-t0))

    def generate_data(self, max_len, batch_size, in_features=240, use_dummy=True):
        import torch
        import intel_pytorch_extension as ipex
        if use_dummy:
            features = torch.rand([max_len, batch_size, in_features]).to(ipex.DEVICE)
            features_lens = torch.rand([batch_size]).to(ipex.DEVICE)
        else:
            features = None
            features_lens = None
        return features, features_lens

    def predict(self, query_idx_list, conf=None, run_mode="inference"):
        features, features_lens, dur_fea = self.predict_fea(query_idx_list)
        logits, logits_lens, dur_enc = self.predict_enc(features, features_lens, conf, run_mode)
        transcripts, dur_dec = self.predict_dec(logits, logits_lens, len(query_idx_list))
        return transcripts, dur_fea, dur_enc, dur_dec

    def predict_fea(self, query_idx_list):
        import torch
        import intel_pytorch_extension as ipex
        t0 = time.time()
        query_len = len(query_idx_list)
        with torch.no_grad():
            """
            if self.num_cores == 1:
                serial_audio_processor = True
            else:
                serial_audio_processor = False
            """
            serial_audio_processor = True
            if serial_audio_processor:
                feature_list = []
                feature_length_list = []
                for idx in query_idx_list:
                    waveform = self.qsl[idx]
                    feature_element, feature_length = self.audio_preprocessor.forward(
                                                            (torch.from_numpy(waveform).unsqueeze(0),
                                                             torch.tensor(len(waveform)).unsqueeze(0)))
                    feature_list.append(feature_element.squeeze(0).transpose_(0, 1))
                    feature_length_list.append(feature_length.squeeze(0))
                feature = torch.nn.utils.rnn.pad_sequence(feature_list, batch_first=True)
                feature_length = torch.tensor(feature_length_list)
            else:
                waveform_list = []
                for idx in query_idx_list:
                    waveform = self.qsl[idx]
                    waveform_list.append(torch.from_numpy(waveform))
                waveform_batch = torch.nn.utils.rnn.pad_sequence(waveform_list, batch_first=True)
                waveform_lengths = torch.tensor([waveform.shape[0] for waveform in waveform_list],
                                                                dtype=torch.int64)
                feature, feature_length = self.audio_preprocessor.forward((waveform_batch, waveform_lengths))

            assert feature.ndim == 3
            assert feature_length.ndim == 1
            local_int8 = self.int8
            local_bf16 = self.bf16
            if query_len == 1:
                local_bf16 = False
            # RNNT can run in the following precision combinations:
            # encoder       | decoder   | --bf16    | --int8
            # --------------+-----------+-----------+---------
            # FP32          | FP32      | False     | False
            # BF16          | BF16      | True      | False
            # INT8          | BF16      | True      | True
            # INT8          | FP32      | False     | True
            if local_bf16 and not local_int8:
                # set bf16 mode globally for both encoder and decoder
                ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
            ipex.core.enable_auto_dnnl()
            feature = feature.to(ipex.DEVICE)
            feature_length = feature_length.to(ipex.DEVICE)
            if serial_audio_processor:
                feature_ = feature.permute(1, 0, 2)
            else:
                feature_ = feature.permute(2, 0, 1)
        t1 = time.time()
        return feature_, feature_length, t1-t0

    def predict_enc(self, feature, feature_len, conf=None, run_mode="inference"):
        import torch
        with torch.no_grad():
            t0 = time.time()
            logits, logits_lens = self.greedy_decoder.forward_enc_batch(feature, feature_len, conf, self.int8, run_mode)
            t1 = time.time()
        return logits, logits_lens, t1-t0

    def predict_dec(self, logits, logits_lens, query_len):
        import torch
        import intel_pytorch_extension as ipex
        t0 = time.time()
        with torch.no_grad():
            local_int8 = self.int8
            local_bf16 = self.bf16
            if query_len == 1:
                local_bf16 = False  # for small workload
            # RNNT can run in the following precision combinations:
            # encoder       | decoder   | --bf16    | --int8
            # --------------+-----------+-----------+---------
            # FP32          | FP32      | False     | False
            # BF16          | BF16      | True      | False
            # INT8          | BF16      | True      | True
            # INT8          | FP32      | False     | True
            if local_bf16 and not local_int8:
                # set bf16 mode globally for both encoder and decoder
                ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
            ipex.core.enable_auto_dnnl()
            if query_len == 1:
                transcripts = self.greedy_decoder.forward_dec_single_batch(logits, logits_lens, local_int8, local_bf16)
            else:
                transcripts = self.greedy_decoder.forward_dec_batch(logits, logits_lens, local_int8, local_bf16)
        t1 = time.time()
        return transcripts, t1-t0

    def run_queue(self):
        import torch
        import intel_pytorch_extension as ipex
        dur_fea, dur_enc, dur_dec = 0, 0, 0
        self.queue_count += 1
        if self.rank in PROFILE_RANK and self.verbose != '0':
            print("========== Rank {0} Iteration {1} Verbose {2} ==========".format(self.rank, self.queue_count, self.verbose))
        if debug and self.queue_count % 10 == 0:
            size = self.task_queue.qsize()
            if (size > 1):
                print ("{} - queue of rank {} piled up, mode={}, samples already in queue={}".format(self.task_queue_group, self.rank, self.mode, size))
        t0 = time.time()
        next_task = self.task_queue.get()
        t1 = time.time()
        self.queue_wait += t1 - t0
        if next_task is None:
            self.task_queue.task_done()
            self.wait_for_none_count -= 1
            if self.wait_for_none_count <= 0:
                self.result_queue.put(None)
                return False
            else:
                return True

        if isinstance(next_task, AddNone):
            self.wait_for_none_count += 1
            return True

        query_id_list = next_task.query_id_list
        query_idx_list = next_task.query_idx_list
        query_len = len(query_idx_list)

        if self.int8:
            conf = ipex.AmpConf(torch.int8, self.configure_path)

        if self.mode=='enc':
            features, features_lens, dur_fea = self.predict_fea(query_idx_list)
            logits, logits_lens, dur_enc = self.predict_enc(features, features_lens, conf)
            self.result_queue.put(EncodeOutput(query_id_list, query_idx_list, logits, logits_lens, dur_fea, dur_enc))

        elif self.mode=='enc+dec':
            if self.rank in PROFILE_RANK and self.profile == "Split":
                from torch.autograd.profiler import profile
                with profile(record_shapes=True) as prof_fea:
                    features, features_lens, dur_fea = self.predict_fea(query_idx_list)
                self.prof_fea_list.append(prof_fea)
                with profile(record_shapes=True) as prof_enc:
                    logits, logits_lens, dur_enc = self.predict_enc(features, features_lens, conf)
                self.prof_enc_list.append(prof_enc)
                with profile(record_shapes=True) as prof_dec:
                    transcripts, dur_dec = self.predict_dec(logits, logits_lens, query_len)
                self.prof_dec_list.append(prof_dec)
            else:
                transcripts, dur_fea, dur_enc, dur_dec = self.predict(query_idx_list, conf)
            assert len(transcripts) == query_len
            for id, trans in zip(query_id_list, transcripts):
                self.result_queue.put(Output(id, trans))

        elif self.mode=='dec':
            logits = next_task.logits
            logits_lens = next_task.logits_lens
            dur_fea = next_task.dur_fea
            dur_enc = next_task.dur_enc
            transcripts, dur_dec = self.predict_dec(logits, logits_lens, query_len)
            assert len(transcripts) == query_len
            for id, trans in zip(query_id_list, transcripts):
                self.result_queue.put(Output(id, trans))

        self.consumer_fea += dur_fea
        self.consumer_enc += dur_enc
        self.consumer_dec += dur_dec
        self.task_queue.task_done()
        return True

    def init_model(self):
        import torch
        from decoders import ScriptGreedyDecoder
        from model_separable_rnnt import RNNT
        from preprocessing import AudioPreprocessing
        if self.model_init:
            return

        config = toml.load(self.config_toml)
        dataset_vocab = config['labels']['labels']
        rnnt_vocab = add_blank_label(dataset_vocab)
        featurizer_config = config['input_eval']
        self.audio_preprocessor = AudioPreprocessing(**featurizer_config)
        self.audio_preprocessor.eval()
        self.audio_preprocessor = torch.jit.script(self.audio_preprocessor)
        self.audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
            torch._C._freeze_module(self.audio_preprocessor._c))

        model = RNNT(
            feature_config=featurizer_config,
            rnnt=config['rnnt'],
            num_classes=len(rnnt_vocab)
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        migrated_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            key = key.replace("joint_net", "joint.net")
            migrated_state_dict[key] = value
        del migrated_state_dict["audio_preprocessor.featurizer.fb"]
        del migrated_state_dict["audio_preprocessor.featurizer.window"]
        model.load_state_dict(migrated_state_dict, strict=True)

        import intel_pytorch_extension as ipex
        if self.bf16:
            ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        ipex.core.enable_auto_dnnl()
        model = model.to(ipex.DEVICE)

        model.eval()
        """
        if not self.ipex:
            model.encoder = torch.jit.script(model.encoder)
            model.encoder = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(model.encoder._c))
            model.prediction = torch.jit.script(model.prediction)
            model.prediction = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(model.prediction._c))
        """
        model.joint = torch.jit.script(model.joint)
        model.joint = torch.jit._recursive.wrap_cpp_module(
            torch._C._freeze_module(model.joint._c))
        """
        if not self.ipex:
            model = torch.jit.script(model)
        """

        self.greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)

        self.model_init = True

    def run(self):
        if self.rank in PROFILE_RANK and self.verbose:
            os.environ['MKLDNN_VERBOSE'] = self.verbose
            print('### set rank={0} MKLDNN_VERBOSE={1}'.format(self.rank, os.getenv('MKLDNN_VERBOSE')))
        print("### ({}) set rank {} to cores {}; omp num threads = {}"
            .format(self.mode, self.rank, self.core_list, self.num_cores))
        str_core_list='{}'.format(self.core_list).replace(' ','').replace('[','').replace(']','')
        os.environ['OMP_NUM_THREADS'] = '{}'.format(self.num_cores)
        os.environ['KMP_AFFINITY'] = 'explicit,proclist=[{}]'.format(str_core_list)
        os.sched_setaffinity(self.pid, self.core_list)
        #cmd = "taskset -p -c %s %d" % (str_core_list, self.pid)
        #print (cmd)
        #os.system(cmd)
        #print ("rank-{} OMP_NUM_THREADS={}".format(self.rank, os.environ['OMP_NUM_THREADS']))
        #print ("rank-{} KMP_AFFINITY={}".format(self.rank, os.environ['KMP_AFFINITY']))

        import torch
        torch.set_num_threads(self.num_cores)
        torch.set_num_interop_threads(1)

        self.init_model()

        self.do_warmup()

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        if self.rank in PROFILE_RANK and self.profile == "True":
            from torch.autograd.profiler import profile
            with profile(record_shapes=True) as prof:
                result = True
                while result:
                    result = self.run_queue()
            while result:
                result = self.run_queue()
        else:
            while self.run_queue():
                pass
        self.lock.acquire()
        self.total_fea.value += self.consumer_fea
        self.total_enc.value += self.consumer_enc
        self.total_dec.value += self.consumer_dec
        self.lock.release()
        if self.rank in PROFILE_RANK:
            if self.profile == 'Split':
                for i in range(len(self.prof_fea_list)):
                    iter = i + 1
                    print('========== Rank {0} Iteration {1} Preprocessing Profile =========='.format(self.rank, iter))
                    trace_handler(self.prof_fea_list[i], 'fea_rank{0}_iter{1}_trace.json'.format(self.rank, iter))
                    print('========== Rank {0} Iteration {1} Encoder Profile =========='.format(self.rank, iter))
                    trace_handler(self.prof_enc_list[i], 'enc_rank{0}_iter{1}_trace.json'.format(self.rank, iter))
                    print('========== Rank {0} Iteration {1} Decoder Profile =========='.format(self.rank, iter))
                    trace_handler(self.prof_dec_list[i], 'dec_rank{0}_iter{1}_trace.json'.format(self.rank, iter))
            if self.profile == 'True':
                print('========== RNNT Profile ==========')
                trace_handler(prof, 'rnnt_rank{0}_trace.json'.format(self.rank))

global t_start
def response_loadgen(out_queue):
    global finish_count
    global query_count
    global t_start
    out_queue_cnt = 0
    os.sched_setaffinity(os.getpid(), [0])
    while True:
        next_task = out_queue.get()
        if next_task is None:
            print("Exiting response thread")
            break

        if isinstance(next_task, AddNone):
            continue

        query_id = next_task.query_id
        transcript = next_task.transcript
        response_array = array.array('q', transcript)
        bi = response_array.buffer_info()
        response = lg.QuerySampleResponse(query_id, bi[0],
                                          bi[1] * response_array.itemsize)
        lg.QuerySamplesComplete([response])
        out_queue_cnt += 1
        finish_count += 1
        if debug:
            if finish_count == 1:
                t_start = time.time()
                print("Finish {} of {} samples".format(finish_count, query_count), end='\r')
            else:
                elapsed = time.time() - t_start
                rate = elapsed/(finish_count - 1)
                remaining_time = (query_count - finish_count)*rate
                print("Finish {} of {} samples, remaining {} seconds.".format(finish_count, query_count, int(remaining_time)), end='\r')

    print("Finish processing {} samples in this queue".format(out_queue_cnt))


class PytorchSUT:
    def __init__(self, config_toml, checkpoint_path, dataset_dir, manifest_filepath,
                 perf_count, machine_conf, enable_debug=False, profile=False, verbose="0",
                 bf16=False, warmup=False, int8=False, configure_path=""):
        ### multi instance attributes
        self.num_cores, self.cores_per_socket, self.num_sockets = get_num_cores()
        self.lock = mp.Lock()
        self.init_counter = mp.Value("i", 0)
        self.total_fea = mp.Value("f", 0)
        self.total_enc = mp.Value("f", 0)
        self.total_dec = mp.Value("f", 0)
        self.output_queue = mp.Queue()
        self.input_queue = mp.JoinableQueue()
        self.decode_queue = mp.JoinableQueue()
        self.bf16 = bf16
        self.int8 = int8
        self.configure_path = configure_path
        self.warmup = warmup

        #server-specific
        self.num_queues = None
        self.core_count_list = []
        self.num_instance_list = []
        self.seq_cutoff_list = []
        self.batch_size_list = []
        self.run_mode= []
        self.input_queue_list = []

        self.read_machine_conf(machine_conf)
        # create queue list
        for _ in range(self.num_queues):
            self.input_queue_list.append(mp.JoinableQueue())

        config = toml.load(config_toml)

        dataset_vocab = config['labels']['labels']
        featurizer_config = config['input_eval']

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries,
                                   self.process_latencies)
        self.qsl = AudioQSLInMemory(dataset_dir,
                                    manifest_filepath,
                                    dataset_vocab,
                                    featurizer_config["sample_rate"],
                                    perf_count)

        self.issue_queue = InQueue(self.input_queue_list, self.qsl,
                                         self.seq_cutoff_list, self.batch_size_list)

        ### worker process
        self.consumers = []
        self.decoders = []
        cur_core_idx = self.cores_for_loadgen
        rank = 0
        for i in range(self.decoder_num_instances):
            self.decoders.append(
                Consumer(self.decode_queue, -1, self.output_queue, self.lock, self.init_counter, rank,
                        [j for j in range(cur_core_idx, cur_core_idx+self.cores_for_decoder)],
                        self.qsl, config_toml, checkpoint_path, dataset_dir, manifest_filepath,
                        perf_count, profile, int8, bf16, warmup, configure_path, 'dec',
                        self.total_fea, self.total_enc, self.total_dec, verbose))
            rank += 1
            cur_core_idx += self.cores_for_decoder
        start_cores = [cur_core_idx]+[0]*(self.num_sockets-1)
        cur_socket = 0
        for i in range(self.num_queues-1, -1, -1):
            curr_cores_per_instance = self.core_count_list[i]
            for _ in range(self.num_instance_list[i]):
                while (start_cores[cur_socket] + curr_cores_per_instance > self.cores_per_socket):
                    cur_socket = (cur_socket+1) % self.num_sockets
                cur_core_idx = start_cores[cur_socket] + cur_socket*self.cores_per_socket
                #print ("assign instance from queue {} to core [{}:{}]".format(i, cur_core_idx, cur_core_idx+curr_cores_per_instance-1))
                self.consumers.append(
                    Consumer(self.input_queue_list[i], i,
                            self.decode_queue if self.run_mode[i]=='enc' else self.output_queue,
                            self.lock, self.init_counter, rank,
                            [i for i in range(cur_core_idx, cur_core_idx + curr_cores_per_instance)],
                            self.qsl, config_toml, checkpoint_path, dataset_dir, manifest_filepath,
                            perf_count, profile, int8, bf16, warmup, configure_path, self.run_mode[i],
                            self.total_fea, self.total_enc, self.total_dec, verbose))
                rank += 1
                start_cores[cur_socket] += curr_cores_per_instance
                cur_socket = (cur_socket+1) % self.num_sockets
        self.num_instances = len(self.consumers) + len(self.decoders)

        ### start worker process
        for d in self.decoders:
            d.start()
        for c in self.consumers:
            c.start()

        ### wait until all sub processes are ready
        block_until(self.init_counter, self.num_instances, 2)

        ### start response thread
        self.response_worker = threading.Thread(
            target=response_loadgen, args=(self.output_queue,))
        self.response_worker.daemon = True
        self.response_worker.start()

        ### debug
        global debug
        debug = enable_debug


    def read_machine_conf(self, machine_conf):

        # machine conf format: type, core_per_instance, num_instances, seq_len_cutoff
        # assuming seq_len_cutoff in increasing order
        infile = open(machine_conf, "r")
        data = infile.read().splitlines()

        self.decoder_num_instances = 0
        for d in data:
            if d[0]=='#':
                continue
            entry_type = d.split()[0]
            values = d.split()[1:]
            if entry_type=='enc' or entry_type=='enc+dec':
                core_count, num_instance, cutoff, batch_size = map(int, values)
                self.core_count_list.append(core_count)
                self.num_instance_list.append(num_instance)
                self.seq_cutoff_list.append(cutoff)
                self.batch_size_list.append(batch_size)
                self.run_mode.append(entry_type)
            if entry_type=='lg':
                core_count, _ = map(int, values)
                self.cores_for_loadgen = core_count
            if entry_type=='dec':
                core_count, num_instance = map(int, values)
                self.cores_for_decoder = core_count
                self.decoder_num_instances = num_instance
        self.num_queues = len(self.core_count_list)
        infile.close()
        #TO DO: validate config

    def issue_queries(self, query_samples):
        global start_time
        global query_count
        if len(query_samples) != 1:
            ### make sure samples in the same batch are about the same length
            # qsl must be reversed sorted for best performance
            query_samples.sort(key=lambda k: self.qsl[k.index].shape[0], reverse=True)
        self.issue_queue.put(query_samples)
        end_time = time.time()
        dur = end_time - start_time
        start_time = end_time
        query_count += len(query_samples)
        if debug:
            print('\n#### issue {} samples in {:.3f} sec: total {} samples'.format(len(query_samples), dur, query_count))

    def flush_queries(self):
        self.issue_queue.put(None)

    def process_latencies(self, latencies_ns):
        print("Average latency (ms) per query:")
        print(np.mean(latencies_ns)/1000000.0)
        print("Median latency (ms): ")
        print(np.percentile(latencies_ns, 50)/1000000.0)
        print("90 percentile latency (ms): ")
        print(np.percentile(latencies_ns, 90)/1000000.0)

    def cal_split_latencies(self):
        avg_fea = self.total_fea.value / self.num_instances
        avg_enc = self.total_enc.value / self.num_instances
        avg_dec = self.total_dec.value / self.num_instances
        avg_rnnt = avg_fea + avg_enc + avg_dec
        print('================================================')
        print('Split Latencies Summary')
        print('================================================')
        print('Preprocess average latency (s) per instance\t: {0:.5f}\t{1:.3f}%'.format(avg_fea, avg_fea/avg_rnnt*100))
        print('Encoder average latency (s) per instance\t: {0:.5f}\t{1:.3f}%'.format(avg_enc, avg_enc/avg_rnnt*100))
        print('Decoder average latency (s) per instance\t: {0:.5f}\t{1:.3f}%'.format(avg_dec, avg_dec/avg_rnnt*100))
        print('RNNT average latency (s) per instance\t: {0:.5f}'.format(avg_rnnt))

    def __del__(self):
        ### clear up sub processes
        for i in range(self.num_queues):
            self.input_queue_list[i].join()
            for _ in range(self.num_instance_list[i]):
                self.input_queue_list[i].put(None)

        for c in self.consumers:
            c.join()

        for i in range(len(self.decoders)):
            self.decode_queue.put(None)
        for d in self.decoders:
            d.join()

        self.output_queue.put(None)
        self.cal_split_latencies()
        print("Finished destroying SUT.")
