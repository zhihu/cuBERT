//
// Created by 田露 on 2019/1/14.
//

#include "cuBERT.h"
#include <random>
#include <chrono>
#include <iostream>


void random_input(int* input_ids, char* input_mask, char* segment_ids, size_t length) {
    std::random_device r;
    std::default_random_engine e(r());

    std::uniform_int_distribution<int> id_dist(0, 21120);
    std::uniform_int_distribution<int> zo_dist(0, 1);
    for (int i = 0; i < length; ++i) {
        input_ids[i] = id_dist(e);
        input_mask[i] = zo_dist(e);
        segment_ids[i] = zo_dist(e);
    }
}

int main() {
    int max_batch_size = 512;
    int batch_size = 400;
    int seq_length = 32;

    int input_ids[batch_size * seq_length];
    char input_mask[batch_size * seq_length];
    char segment_ids[batch_size * seq_length];

    float logits[batch_size];

    void* model = cuBERT_open("bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12);

    // warm up
    for (int i = 0; i < 10; ++i) {
        random_input(input_ids, input_mask, segment_ids, batch_size * seq_length);

        auto start = std::chrono::high_resolution_clock::now();
        cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, logits);
        auto finish = std::chrono::high_resolution_clock::now();

        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "warm_up: " << milli << "ms" << std::endl;
    }

    // benchmark
    for (int i = 0; i < 10; ++i) {
        random_input(input_ids, input_mask, segment_ids, batch_size * seq_length);

        auto start = std::chrono::high_resolution_clock::now();
        cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, logits);
        auto finish = std::chrono::high_resolution_clock::now();

        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "benchmark: " << milli << "ms" << std::endl;
    }

    cuBERT_close(model);
}
