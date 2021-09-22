# Copyright 2020 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import torch
import toml
from QSL import AudioQSLInMemory
from pytorch_SUT import Consumer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_config_toml", default="pytorch/configs/rnnt.toml")
    parser.add_argument("--pytorch_checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--perf_count", type=int, default=None)
    parser.add_argument("--configure_save_path", required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    config = toml.load(args.pytorch_config_toml)
    dataset_vocab = config['labels']['labels']
    featurizer_config = config['input_eval']

    qsl = AudioQSLInMemory(args.dataset_dir,
                           args.manifest,
                           dataset_vocab,
                           featurizer_config['sample_rate'],
                           args.perf_count)

    consumer = Consumer(None, None, None, None, 0, 0, [0], qsl, args.pytorch_config_toml, args.pytorch_checkpoint,
                        args.dataset_dir, args.manifest, None, False, True, True, False, None, 'enc+dec')

    consumer.init_model()

    print ("QSL has {} samples".format(qsl.count))

    # start calibration
    import intel_pytorch_extension as ipex
    conf = ipex.AmpConf(torch.int8)

    print ("start prediction")
    for i in range(0, qsl.count, 32):
        print ("calibrate sample in range ({}, {})".format(i, min(i+32, qsl.count)))
        consumer.predict(range(i, min(i+32, qsl.count)), conf, run_mode="calibration")
    print ("start prediction end")

    conf.save(args.configure_save_path)

    print("Done!")


if __name__ == "__main__":
    main()
