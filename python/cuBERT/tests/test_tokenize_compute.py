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
from cuBERT import cuBERT_close_tokenizer
from cuBERT import cuBERT_ComputeType
from cuBERT import cuBERT_open
from cuBERT import cuBERT_open_tokenizer
from cuBERT import cuBERT_OutputType
from cuBERT import cuBERT_tokenize_compute

if getattr(sys, 'frozen', False):
    cur_path = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    cur_path = os.path.dirname(os.path.abspath(__file__))
else:
    cur_path = ''


class TestTokenizeCompute(unittest.TestCase):
    max_batch_size = 128
    batch_size = 2
    hidden_size = 768
    seq_length = 32

    str_txt_a = ['知乎', '知乎']
    str_txt_b = ['在家刷知乎', '知乎发现更大的世界']

    unc_txt_a = [u'知乎', u'知乎']
    unc_txt_b = [u'在家刷知乎', u'知乎发现更大的世界']

    model_filename = os.path.join(cur_path, 'bert_frozen_seq32.pb')
    vocab_filename = os.path.join(cur_path, 'vocab.txt')

    # pylint: disable=too-many-arguments
    def compute(self, text_a, text_b, output_type, compute_type, model, tokenizer):
        return cuBERT_tokenize_compute(
            self.batch_size, text_a, text_b, output_type, compute_type,
            model, tokenizer, self.hidden_size, self.seq_length
        )  # yapf: disable

    def test_compute_logits_float(self):
        model = cuBERT_open(self.model_filename, self.max_batch_size, self.seq_length, 12, 12)
        tokenizer = cuBERT_open_tokenizer(self.vocab_filename)

        opt = cuBERT_OutputType.cuBERT_LOGITS
        ct = cuBERT_ComputeType.cuBERT_COMPUTE_FLOAT  # pylint: disable=invalid-name

        self.assertListEqual(
            [-2.51366, - 1.47348],
            list([round(value, 5) for value in self.compute(self.str_txt_a, self.str_txt_b, opt, ct, model, tokenizer)])
        )  # yapf: disable

        self.assertListEqual(
            [-2.51366, - 1.47348],
            list([round(value, 5) for value in self.compute(self.str_txt_a, self.unc_txt_b, opt, ct, model, tokenizer)])
        )  # yapf: disable

        self.assertListEqual(
            [-2.51366, - 1.47348],
            list([round(value, 5) for value in self.compute(self.unc_txt_a, self.str_txt_b, opt, ct, model, tokenizer)])
        )  # yapf: disable

        self.assertListEqual(
            [-2.51366, - 1.47348],
            list([round(value, 5) for value in self.compute(self.unc_txt_a, self.unc_txt_b, opt, ct, model, tokenizer)])
        )  # yapf: disable

        cuBERT_close_tokenizer(tokenizer)
        cuBERT_close(model)

    def test_compute_logits_half(self):
        cuda_support = True
        try:
            subprocess.check_output(['nvcc', '--version'])
        except OSError:
            cuda_support = False  # 'no cuda supported, using MKL'

        model = cuBERT_open(self.model_filename, self.max_batch_size, self.seq_length, 12, 12)
        tokenizer = cuBERT_open_tokenizer(self.vocab_filename)

        if not cuda_support:
            self.assertRaises(ValueError)
        else:
            pass  # todo: test on GPU

        cuBERT_close_tokenizer(tokenizer)
        cuBERT_close(model)


if __name__ == '__main__':
    unittest.main()
