# -*- coding: UTF-8 -*-
import time
import numpy as np
import libcubert as cuBERT

max_batch_size = 128
batch_size = 128
seq_length = 32
hidden_size = 768

output_size = batch_size * hidden_size
output_type=cuBERT.OutputType.POOLED_OUTPUT
compute_type=cuBERT.ComputeType.FLOAT

input_ids = np.random.randint(0, 21120, size=(batch_size, seq_length), dtype=np.int32)
input_mask = np.random.randint(0, 1, size=(batch_size, seq_length), dtype=np.int8)
segment_ids = np.random.randint(0, 1, size=(batch_size, seq_length), dtype=np.int8)
output = np.zeros([batch_size, hidden_size], dtype=np.float32, order='C')

def benchmark(model):
    start = time.time() * 1000
    model.compute(input_ids, input_mask, segment_ids, output, 
                  output_type=output_type)
    finish = time.time() * 1000
    milli = finish - start
    print("cuBERT: {} ms".format(milli))

if __name__ == '__main__':
    model = cuBERT.Model("../build/bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12, compute_type=compute_type)
    print("=== warm_up ===")
    for i in range(10):
        benchmark(model)
    print("=== benchmark ===")
    for i in range(10):
        benchmark(model)
