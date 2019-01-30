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

// KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0
int main() {
    std::default_random_engine e(0);

    int max_batch_size = 4;
    int batch_size = 1;
    int seq_length = 32;

    // mklBERT
    int input_ids[batch_size * seq_length];
    char input_mask[batch_size * seq_length];
    char segment_ids[batch_size * seq_length];

    void* model = mklBERT_open("bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12);
    std::ofstream result("mklBERT.txt");

    std::cout << "=== warm_up ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        random_input(e, input_ids, input_mask, segment_ids, batch_size * seq_length);

        auto start = std::chrono::high_resolution_clock::now();
        float* logits = mklBERT_compute(model, batch_size, input_ids, input_mask, segment_ids);
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "mklBERT: " << milli << "ms" << std::endl;

        for (int j = 0; j < batch_size; ++j) {
            result << logits[j] << std::endl;
        }
    }

    std::cout << "=== benchmark ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        random_input(e, input_ids, input_mask, segment_ids, batch_size * seq_length);

        auto start = std::chrono::high_resolution_clock::now();
        float* logits = mklBERT_compute(model, batch_size, input_ids, input_mask, segment_ids);
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "mklBERT: " << milli << "ms" << std::endl;

        for (int j = 0; j < batch_size; ++j) {
            result << logits[j] << std::endl;
        }
    }

    result.close();
    mklBERT_close(model);
}
