#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <fstream>

#include "cuBERT.h"

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

int main() {
    std::default_random_engine e(0);

    int max_batch_size = 128;
    int batch_size = 128;
    int seq_length = 32;

    // cuBERT
    int input_ids[batch_size * seq_length];
    char input_mask[batch_size * seq_length];
    char segment_ids[batch_size * seq_length];
    float logits[batch_size];

    void* model = cuBERT_open("bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12);
    std::ofstream result("cuBERT.txt");

    std::cout << "=== warm_up ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        random_input(e, input_ids, input_mask, segment_ids, batch_size * seq_length);

        auto start = std::chrono::high_resolution_clock::now();
        cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, logits);
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "cuBERT: " << milli << "ms" << std::endl;

        for (int j = 0; j < batch_size; ++j) {
            result << logits[j] << std::endl;
        }
    }

    std::cout << "=== benchmark ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        random_input(e, input_ids, input_mask, segment_ids, batch_size * seq_length);

        auto start = std::chrono::high_resolution_clock::now();
        cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, logits);
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "cuBERT: " << milli << "ms" << std::endl;

        for (int j = 0; j < batch_size; ++j) {
            result << logits[j] << std::endl;
        }
    }

    result.close();
    cuBERT_close(model);
}
