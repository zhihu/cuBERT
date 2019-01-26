#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuBERT/bert/BertBatchMatMul.h"
using namespace cuBERT;

class BertBatchMatMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        cublasCreate_v2(&handle);
    }

    void TearDown() override {
        cublasDestroy_v2(handle);
    }

    cublasHandle_t handle;
};

TEST_F(BertBatchMatMulTest, qk) {
    size_t seq_length = 2;
    size_t num_attention_heads = 4;
    size_t size_per_head = 3;

    float query[24] = {
            0, 1, 2,
            -1, -2, 0,
            2, -1, 0,
            0, 1, -2,

            2, 1, -2,
            0, 2, -1,
            -1, -1, 0,
            2, 1, -1,
    };
    float key[24] = {
            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,
    };
    float out[16];

    float *query_gpu;
    float *key_gpu;
    float *out_gpu;
    cudaMalloc(&query_gpu, sizeof(float) * 24);
    cudaMalloc(&key_gpu, sizeof(float) * 24);
    cudaMalloc(&out_gpu, sizeof(float) * 16);

    cudaMemcpy(query_gpu, query, sizeof(float) * 24, cudaMemcpyHostToDevice);
    cudaMemcpy(key_gpu, key, sizeof(float) * 24, cudaMemcpyHostToDevice);

    BertQK bbmm(handle, seq_length, num_attention_heads, size_per_head);
    bbmm.compute(1, query_gpu, key_gpu, out_gpu);

    cudaMemcpy(out, out_gpu, sizeof(float) * 16, cudaMemcpyDeviceToHost);

    cudaFree(query_gpu);
    cudaFree(key_gpu);
    cudaFree(out_gpu);

    EXPECT_FLOAT_EQ(out[0], -2);
    EXPECT_FLOAT_EQ(out[1], 3);
    EXPECT_FLOAT_EQ(out[2], -8);
    EXPECT_FLOAT_EQ(out[3], -3);
    EXPECT_FLOAT_EQ(out[4], 1);
    EXPECT_FLOAT_EQ(out[5], 4);
    EXPECT_FLOAT_EQ(out[6], -1);
    EXPECT_FLOAT_EQ(out[7], 0);
    EXPECT_FLOAT_EQ(out[8], 8);
    EXPECT_FLOAT_EQ(out[9], -1);
    EXPECT_FLOAT_EQ(out[10], 9);
    EXPECT_FLOAT_EQ(out[11], 2);
    EXPECT_FLOAT_EQ(out[12], 1);
    EXPECT_FLOAT_EQ(out[13], -2);
    EXPECT_FLOAT_EQ(out[14], 2);
    EXPECT_FLOAT_EQ(out[15], 0);
}
