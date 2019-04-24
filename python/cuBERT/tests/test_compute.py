#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2019 R&D, NTC Inc. (ntc.ai)
#
# Author: qinluo <eric.x.sun@gmail.com>
#

import os
import subprocess
import sys
import unittest

from cuBERT import cuBERT_close
from cuBERT import cuBERT_compute
from cuBERT import cuBERT_ComputeType
from cuBERT import cuBERT_open
from cuBERT import cuBERT_OutputType

if getattr(sys, 'frozen', False):
    cur_path = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    cur_path = os.path.dirname(os.path.abspath(__file__))
else:
    cur_path = ''


class TestCompute(unittest.TestCase):
    max_batch_size = 10
    batch_size = 3
    hidden_size = 768
    seq_length = 10

    input_ids = [
        4176, 8169, 14426, 9893, 77, 18766, 17309, 16372, 8547, 4473, 19231, 12151, 18055, 7358, 9575, 1298, 15753,
        19021, 12694, 5114, 20481, 16099, 20135, 13092, 4998, 8605, 447, 7301, 5139, 5745
    ]

    input_mask = [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]

    segment_ids = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0]

    model_filename = os.path.join(cur_path, 'bert_frozen_seq32.pb')

    def test_compute_logits_float(self):
        model = cuBERT_open(self.model_filename, self.max_batch_size, self.seq_length, 12, 12)

        logits = cuBERT_compute(
            self.batch_size, self.input_ids, self.input_mask, self.segment_ids,
            cuBERT_OutputType.cuBERT_LOGITS, cuBERT_ComputeType.cuBERT_COMPUTE_FLOAT, model
        )  # yapf: disable

        cuBERT_close(model)

        self.assertListEqual([-5.55838, -5.87640, -5.45485], list([round(value, 5) for value in logits]))

    def test_compute_logits_half(self):
        cuda_support = True
        try:
            subprocess.check_output(['nvcc', '--version'])
        except OSError:
            cuda_support = False  # 'no cuda supported, using MKL'

        model = cuBERT_open(self.model_filename, self.max_batch_size, self.seq_length, 12, 12)

        if not cuda_support:
            self.assertRaises(ValueError)
        else:
            pass  # todo: test on GPU

        cuBERT_close(model)


if __name__ == '__main__':
    unittest.main()
