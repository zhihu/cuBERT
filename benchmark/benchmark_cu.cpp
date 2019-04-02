#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <fstream>
#include <half.hpp>

#include "cuBERT.h"

// define half/float precision
// typedef half_float::half Dtype;
// cuBERT_ComputeType compute_type = cuBERT_COMPUTE_HALF;
typedef float Dtype;
cuBERT_ComputeType compute_type = cuBERT_COMPUTE_FLOAT;

const int max_batch_size = 128;
const int batch_size = 128;
const int seq_length = 32;
const int hidden_size = 768;

// define model output
const int output_size = batch_size * hidden_size;
cuBERT_OutputType output_type = cuBERT_POOLED_OUTPUT;

void random_input(std::default_random_engine& e,
                  int* input_ids, char* input_mask, char* segment_ids, size_t length) {
    std::uniform_int_distribution<int> id_dist(0, 21120);
    std::uniform_int_distribution<int> zo_dist(0, 1);
    for (int i = 0; i < length; ++i) {
        input_ids[i] = id_dist(e);
        input_mask[i] = zo_dist(e);
        segment_ids[i] = zo_dist(e);
    }
}

void benchmark(void* model,
               int* input_ids,
               char* input_mask,
               char* segment_ids,
               Dtype* logits,
               std::ofstream& result) {
    auto start = std::chrono::high_resolution_clock::now();
    cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, logits, output_type, compute_type);
    auto finish = std::chrono::high_resolution_clock::now();
    long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "cuBERT: " << milli << "ms" << std::endl;

    for (int j = 0; j < output_size; ++j) {
        result << (float) logits[j] << std::endl;
    }
}

// OMP_NUM_THREADS=? KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0
int main() {
    cuBERT_initialize();

    std::default_random_engine e(0);

    // cuBERT
    int input_ids[batch_size * seq_length];
    char input_mask[batch_size * seq_length];
    char segment_ids[batch_size * seq_length];
    Dtype logits[output_size];

    void* model = cuBERT_open("bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12, compute_type);
    std::ofstream result("cuBERT.txt");

    std::cout << "=== warm_up ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        random_input(e, input_ids, input_mask, segment_ids, batch_size * seq_length);
        benchmark(model, input_ids, input_mask, segment_ids, logits, result);
    }

    std::cout << "=== benchmark ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        random_input(e, input_ids, input_mask, segment_ids, batch_size * seq_length);
        benchmark(model, input_ids, input_mask, segment_ids, logits, result);
    }

    result.close();
    cuBERT_close(model, compute_type);

    cuBERT_finalize();
}
