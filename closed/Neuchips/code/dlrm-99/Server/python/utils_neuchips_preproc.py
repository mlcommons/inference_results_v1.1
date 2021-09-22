#
# NEUCHIPS CONFIDENTIAL
#
# Copyright (C) 2018-2021 NEUCHIPS Corp.
# All Rights Reserved.
# Author: Chiung-Liang Lin <chiungliang_lin@neuchips.ai>
#
# The information and source code contained herein is the exclusive property
# of NEUCHIPS CORPORATION and may not be disclosed, examined or reproduced
# in whole or in part without explicit written authorization from the company.
#
import torch
import numpy as np

def quant_pow2_unsigned_s(x, q_frac):

    return torch.clamp(torch.floor(x.to(torch.float64) * pow(2, q_frac) + 0.5), 0, 255).to(torch.uint8)


def hw_s(s):

    zerox16 = np.zeros(16, dtype=np.uint8)
    res = []

    s_t = s.T
    for inf in s_t:
        for i in range(26):
            res.extend(list(int(inf[i].item()).to_bytes(8, 'little', signed=False)))
        res.extend(np.repeat(zerox16, 3))

    new_s_t = np.asarray(res, dtype=np.uint8)
    new_s_t = np.reshape(new_s_t, (s.shape[1], 256))

    return torch.from_numpy(new_s_t.T)
