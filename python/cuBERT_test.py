# -*- coding: UTF-8 -*-
import numpy as np
import libcubert as cuBERT

max_batch_size = 128
batch_size = 2
seq_length = 32

input_ids = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
], dtype=np.int32)

input_mask = np.array([
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]
], dtype=np.int8)

segment_ids = np.array([
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
], dtype=np.int8)

output_type=cuBERT.OutputType.LOGITS
compute_type=cuBERT.ComputeType.FLOAT
output = np.zeros([batch_size], dtype=np.float32, order='C')

model = cuBERT.Model("../build/bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12, 
                     compute_type=compute_type)
model.compute(input_ids, input_mask, segment_ids, output, output_type=output_type)
np.testing.assert_almost_equal([-2.9427543, -1.4876306], output, 5)


text_a = [u"知乎", u"知乎"]
text_b = [u"在家刷知乎", u"知乎发现更大的世界"]
model = cuBERT.Model("../build/bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12, 
                     compute_type=compute_type,
                     vocab_file="../build/vocab.txt")
model.tokenize_compute(text_a, text_b, output, output_type=output_type)
np.testing.assert_almost_equal([-2.51366, -1.47348], output, 5)
