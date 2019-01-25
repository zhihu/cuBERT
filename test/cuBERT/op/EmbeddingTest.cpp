//
// Created by 田露 on 2019/1/22.
//

#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include "cuBERT/op/Embedding.h"
using namespace cuBERT;

class EmbeddingTest : public ::testing::Test {

};

TEST_F(EmbeddingTest, compute) {
    size_t vocab_size = 4;
    size_t embedding_size = 3;

    float embedding_table[12] = {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
    };

    Embedding embedding(vocab_size, embedding_size, embedding_table);

    int input_ids[3] = {2, 2, 1};
    float out[9];

    int* input_ids_gpu;
    float* out_gpu;
    cudaMalloc(&input_ids_gpu, 3 * sizeof(int));
    cudaMalloc(&out_gpu, 9 * sizeof(float));

    cudaMemcpy(input_ids_gpu, input_ids, 3 * sizeof(int), cudaMemcpyHostToDevice);

    embedding.compute(input_ids_gpu, 3, out_gpu, nullptr);

    cudaMemcpy(out, out_gpu, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(out_gpu);
    cudaFree(input_ids_gpu);

    EXPECT_FLOAT_EQ(out[0], 6);
    EXPECT_FLOAT_EQ(out[1], 7);
    EXPECT_FLOAT_EQ(out[2], 8);
    EXPECT_FLOAT_EQ(out[3], 6);
    EXPECT_FLOAT_EQ(out[4], 7);
    EXPECT_FLOAT_EQ(out[5], 8);
    EXPECT_FLOAT_EQ(out[6], 3);
    EXPECT_FLOAT_EQ(out[7], 4);
    EXPECT_FLOAT_EQ(out[8], 5);
}

TEST_F(EmbeddingTest, compute_cpu) {
    size_t vocab_size = 4;
    size_t embedding_size = 3;

    float embedding_table[12] = {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
    };

    Embedding embedding(vocab_size, embedding_size, embedding_table);

    int input_ids[3] = {2, 2, 1};
    float out[9];

    embedding.compute_cpu(input_ids, 3, out);

    EXPECT_FLOAT_EQ(out[0], 6);
    EXPECT_FLOAT_EQ(out[1], 7);
    EXPECT_FLOAT_EQ(out[2], 8);
    EXPECT_FLOAT_EQ(out[3], 6);
    EXPECT_FLOAT_EQ(out[4], 7);
    EXPECT_FLOAT_EQ(out[5], 8);
    EXPECT_FLOAT_EQ(out[6], 3);
    EXPECT_FLOAT_EQ(out[7], 4);
    EXPECT_FLOAT_EQ(out[8], 5);
}