#include "gtest/gtest.h"
#include <random>
#include <iostream>

#include "cuBERT.h"

class cuBertTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT_initialize();
    }

    void TearDown() override {
        cuBERT_finalize();
    }
};

void random_input(int* input_ids, int8_t* input_mask, int8_t* segment_ids, size_t length) {
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
    int max_batch_size = 128;
    int batch_size = 128;
    int seq_length = 32;

    int input_ids[batch_size * seq_length];
    int8_t input_mask[batch_size * seq_length];
    int8_t segment_ids[batch_size * seq_length];
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

TEST_F(cuBertTest, compute_tokenize) {
    int max_batch_size = 128;
    int batch_size = 2;
    int seq_length = 32;
    float output[batch_size];

    const char* text_a[] = {u8"知乎", u8"知乎"};
    const char* text_b[] = {u8"在家刷知乎", u8"知乎发现更大的世界"};

    void* model = cuBERT_open("bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12);
    void* tokenizer = cuBERT_open_tokenizer("vocab.txt");

    cuBERT_tokenize_compute(model, tokenizer, batch_size,
                            text_a,
                            text_b,
                            output, cuBERT_LOGITS);

    cuBERT_close_tokenizer(tokenizer);
    cuBERT_close(model);

    std::cout << output[0] << std::endl;
    std::cout << output[1] << std::endl;
}

#ifdef HAVE_CUDA
TEST_F(cuBertTest, compute_tokenize_mf) {
    int max_batch_size = 128;
    int batch_size = 2;
    int seq_length = 32;
    float logits[batch_size];
    cuBERT_Output output;
    output.logits = logits;

    const char* text_a[] = {u8"知乎", u8"知乎"};
    const char* text_b[] = {u8"在家刷知乎", u8"知乎发现更大的世界"};

    void* model = cuBERT_open("bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12, cuBERT_COMPUTE_HALF);
    void* tokenizer = cuBERT_open_tokenizer("vocab.txt");

    cuBERT_tokenize_compute_m(model, tokenizer, batch_size,
                              text_a,
                              text_b,
                              &output, cuBERT_COMPUTE_HALF, 1);

    cuBERT_close_tokenizer(tokenizer);
    cuBERT_close(model);

    std::cout << logits[0] << std::endl;
    std::cout << logits[1] << std::endl;
}
#endif