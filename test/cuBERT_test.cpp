#include "gtest/gtest.h"
#include <random>
#include <iostream>

#include "../cuBERT.h"

class cuBertTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

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

TEST_F(cuBertTest, compute) {
    int max_batch_size = 512;
    int batch_size = 400;
    int seq_length = 32;

    int input_ids[batch_size * seq_length];
    char input_mask[batch_size * seq_length];
    char segment_ids[batch_size * seq_length];
    float logits[batch_size];
    random_input(input_ids, input_mask, segment_ids, batch_size * seq_length);

    void* model = cuBERT_open("bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12);
    cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, logits);
    cuBERT_close(model);

    std::cout << logits[0] << std::endl;
    std::cout << logits[1] << std::endl;
    std::cout << logits[2] << std::endl;
    std::cout << logits[3] << std::endl;
    std::cout << logits[4] << std::endl;
    std::cout << logits[5] << std::endl;
    std::cout << logits[6] << std::endl;
    std::cout << logits[7] << std::endl;
    std::cout << logits[8] << std::endl;
    std::cout << logits[9] << std::endl;
}
