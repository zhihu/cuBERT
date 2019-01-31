#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include "cuBERT/common.h"
#include "cuBERT/bert/BertBatchMatMul.h"
using namespace cuBERT;

class BertBatchMatMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize();
        handle = cuBERT::blas_create();
    }

    void TearDown() override {
        cuBERT::blas_destroy(handle);
        cuBERT::finalize();
    }

    void* handle;
};

TEST_F(BertBatchMatMulTest, qk) {
    size_t seq_length = 2;
    size_t num_attention_heads = 4;
    size_t size_per_head = 3;

    float *query_gpu;
    float *key_gpu;
    float *out_gpu;
    cudaMalloc(&query_gpu, sizeof(float) * 48);
    cudaMalloc(&key_gpu, sizeof(float) * 48);
    cudaMalloc(&out_gpu, sizeof(float) * 32);
    BertQK bqk(handle, 2, seq_length, num_attention_heads, size_per_head, query_gpu, key_gpu, out_gpu);

    float query[48] = {
            0, 1, 2,
            -1, -2, 0,
            2, -1, 0,
            0, 1, -2,

            2, 1, -2,
            0, 2, -1,
            -1, -1, 0,
            2, 1, -1,

            0, 1, 2,
            -1, -2, 0,
            2, -1, 0,
            0, 1, -2,

            2, 1, -2,
            0, 2, -1,
            -1, -1, 0,
            2, 1, -1,
    };
    float key[48] = {
            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,

            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,
    };
    float out[32];

    cudaMemcpy(query_gpu, query, sizeof(float) * 48, cudaMemcpyHostToDevice);
    cudaMemcpy(key_gpu, key, sizeof(float) * 48, cudaMemcpyHostToDevice);
    bqk.compute(2);
    cudaMemcpy(out, out_gpu, sizeof(float) * 32, cudaMemcpyDeviceToHost);

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
    EXPECT_FLOAT_EQ(out[16], -2);
    EXPECT_FLOAT_EQ(out[17], 3);
    EXPECT_FLOAT_EQ(out[18], -8);
    EXPECT_FLOAT_EQ(out[19], -3);
    EXPECT_FLOAT_EQ(out[20], 1);
    EXPECT_FLOAT_EQ(out[21], 4);
    EXPECT_FLOAT_EQ(out[22], -1);
    EXPECT_FLOAT_EQ(out[23], 0);
    EXPECT_FLOAT_EQ(out[24], 8);
    EXPECT_FLOAT_EQ(out[25], -1);
    EXPECT_FLOAT_EQ(out[26], 9);
    EXPECT_FLOAT_EQ(out[27], 2);
    EXPECT_FLOAT_EQ(out[28], 1);
    EXPECT_FLOAT_EQ(out[29], -2);
    EXPECT_FLOAT_EQ(out[30], 2);
    EXPECT_FLOAT_EQ(out[31], 0);
}

class BertBatchMatMulCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize(true);
    }

    void TearDown() override {
        cuBERT::finalize();
    }
};

TEST_F(BertBatchMatMulCPUTest, qk) {
    size_t seq_length = 2;
    size_t num_attention_heads = 4;
    size_t size_per_head = 3;

    float query[48] = {
            0, 1, 2,
            -1, -2, 0,
            2, -1, 0,
            0, 1, -2,

            2, 1, -2,
            0, 2, -1,
            -1, -1, 0,
            2, 1, -1,

            0, 1, 2,
            -1, -2, 0,
            2, -1, 0,
            0, 1, -2,

            2, 1, -2,
            0, 2, -1,
            -1, -1, 0,
            2, 1, -1,
    };
    float key[48] = {
            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,

            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,
    };
    float out[32];

    BertQK bqk(nullptr, 2, seq_length, num_attention_heads, size_per_head, query, key, out);
    bqk.compute(2);

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
    EXPECT_FLOAT_EQ(out[16], -2);
    EXPECT_FLOAT_EQ(out[17], 3);
    EXPECT_FLOAT_EQ(out[18], -8);
    EXPECT_FLOAT_EQ(out[19], -3);
    EXPECT_FLOAT_EQ(out[20], 1);
    EXPECT_FLOAT_EQ(out[21], 4);
    EXPECT_FLOAT_EQ(out[22], -1);
    EXPECT_FLOAT_EQ(out[23], 0);
    EXPECT_FLOAT_EQ(out[24], 8);
    EXPECT_FLOAT_EQ(out[25], -1);
    EXPECT_FLOAT_EQ(out[26], 9);
    EXPECT_FLOAT_EQ(out[27], 2);
    EXPECT_FLOAT_EQ(out[28], 1);
    EXPECT_FLOAT_EQ(out[29], -2);
    EXPECT_FLOAT_EQ(out[30], 2);
    EXPECT_FLOAT_EQ(out[31], 0);
}


TEST_F(BertBatchMatMulTest, qkv) {
    size_t seq_length = 2;
    size_t num_attention_heads = 4;
    size_t size_per_head = 3;

    float *qk_gpu;
    float *value_gpu;
    float *out_gpu;
    cudaMalloc(&qk_gpu, sizeof(float) * 32);
    cudaMalloc(&value_gpu, sizeof(float) * 48);
    cudaMalloc(&out_gpu, sizeof(float) * 48);
    BertQKV bqkv(handle, 2, seq_length, num_attention_heads, size_per_head, qk_gpu, value_gpu, out_gpu);

    float qk[32] = {
            0, 1,
            -1, -2,
            2, -1,
            0, 1,

            2, 1,
            0, 2,
            -1, -1,
            2, 1,

            0, 1,
            -1, -2,
            2, -1,
            0, 1,

            2, 1,
            0, 2,
            -1, -1,
            2, 1,
    };
    float value[48] = {
            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,

            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,
    };
    float out[48];

    cudaMemcpy(qk_gpu, qk, sizeof(float) * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(value_gpu, value, sizeof(float) * 48, cudaMemcpyHostToDevice);
    bqkv.compute(2);
    cudaMemcpy(out, out_gpu, sizeof(float) * 48, cudaMemcpyDeviceToHost);

    cudaFree(qk_gpu);
    cudaFree(value_gpu);
    cudaFree(out_gpu);
    EXPECT_FLOAT_EQ(out[0], -2);
    EXPECT_FLOAT_EQ(out[1], 3);
    EXPECT_FLOAT_EQ(out[2], 0);
    EXPECT_FLOAT_EQ(out[3], 0);
    EXPECT_FLOAT_EQ(out[4], -7);
    EXPECT_FLOAT_EQ(out[5], -1);
    EXPECT_FLOAT_EQ(out[6], -2);
    EXPECT_FLOAT_EQ(out[7], -2);
    EXPECT_FLOAT_EQ(out[8], -2);
    EXPECT_FLOAT_EQ(out[9], 0);
    EXPECT_FLOAT_EQ(out[10], 0);
    EXPECT_FLOAT_EQ(out[11], 0);
    EXPECT_FLOAT_EQ(out[12], 4);
    EXPECT_FLOAT_EQ(out[13], 3);
    EXPECT_FLOAT_EQ(out[14], -2);
    EXPECT_FLOAT_EQ(out[15], -2);
    EXPECT_FLOAT_EQ(out[16], 4);
    EXPECT_FLOAT_EQ(out[17], 4);
    EXPECT_FLOAT_EQ(out[18], -2);
    EXPECT_FLOAT_EQ(out[19], 1);
    EXPECT_FLOAT_EQ(out[20], 1);
    EXPECT_FLOAT_EQ(out[21], 2);
    EXPECT_FLOAT_EQ(out[22], 2);
    EXPECT_FLOAT_EQ(out[23], 2);
    EXPECT_FLOAT_EQ(out[24], -2);
    EXPECT_FLOAT_EQ(out[25], 3);
    EXPECT_FLOAT_EQ(out[26], 0);
    EXPECT_FLOAT_EQ(out[27], 0);
    EXPECT_FLOAT_EQ(out[28], -7);
    EXPECT_FLOAT_EQ(out[29], -1);
    EXPECT_FLOAT_EQ(out[30], -2);
    EXPECT_FLOAT_EQ(out[31], -2);
    EXPECT_FLOAT_EQ(out[32], -2);
    EXPECT_FLOAT_EQ(out[33], 0);
    EXPECT_FLOAT_EQ(out[34], 0);
    EXPECT_FLOAT_EQ(out[35], 0);
    EXPECT_FLOAT_EQ(out[36], 4);
    EXPECT_FLOAT_EQ(out[37], 3);
    EXPECT_FLOAT_EQ(out[38], -2);
    EXPECT_FLOAT_EQ(out[39], -2);
    EXPECT_FLOAT_EQ(out[40], 4);
    EXPECT_FLOAT_EQ(out[41], 4);
    EXPECT_FLOAT_EQ(out[42], -2);
    EXPECT_FLOAT_EQ(out[43], 1);
    EXPECT_FLOAT_EQ(out[44], 1);
    EXPECT_FLOAT_EQ(out[45], 2);
    EXPECT_FLOAT_EQ(out[46], 2);
    EXPECT_FLOAT_EQ(out[47], 2);
}


TEST_F(BertBatchMatMulCPUTest, qkv_compute) {
    size_t seq_length = 2;
    size_t num_attention_heads = 4;
    size_t size_per_head = 3;

    float qk[32] = {
            0, 1,
            -1, -2,
            2, -1,
            0, 1,

            2, 1,
            0, 2,
            -1, -1,
            2, 1,

            0, 1,
            -1, -2,
            2, -1,
            0, 1,

            2, 1,
            0, 2,
            -1, -1,
            2, 1,
    };
    float value[48] = {
            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,

            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,
    };
    float out[48];

    BertQKV bqkv(nullptr, 2, seq_length, num_attention_heads, size_per_head, qk, value, out);
    bqkv.compute(2);

    EXPECT_FLOAT_EQ(out[0], -2);
    EXPECT_FLOAT_EQ(out[1], 3);
    EXPECT_FLOAT_EQ(out[2], 0);
    EXPECT_FLOAT_EQ(out[3], 0);
    EXPECT_FLOAT_EQ(out[4], -7);
    EXPECT_FLOAT_EQ(out[5], -1);
    EXPECT_FLOAT_EQ(out[6], -2);
    EXPECT_FLOAT_EQ(out[7], -2);
    EXPECT_FLOAT_EQ(out[8], -2);
    EXPECT_FLOAT_EQ(out[9], 0);
    EXPECT_FLOAT_EQ(out[10], 0);
    EXPECT_FLOAT_EQ(out[11], 0);
    EXPECT_FLOAT_EQ(out[12], 4);
    EXPECT_FLOAT_EQ(out[13], 3);
    EXPECT_FLOAT_EQ(out[14], -2);
    EXPECT_FLOAT_EQ(out[15], -2);
    EXPECT_FLOAT_EQ(out[16], 4);
    EXPECT_FLOAT_EQ(out[17], 4);
    EXPECT_FLOAT_EQ(out[18], -2);
    EXPECT_FLOAT_EQ(out[19], 1);
    EXPECT_FLOAT_EQ(out[20], 1);
    EXPECT_FLOAT_EQ(out[21], 2);
    EXPECT_FLOAT_EQ(out[22], 2);
    EXPECT_FLOAT_EQ(out[23], 2);
    EXPECT_FLOAT_EQ(out[24], -2);
    EXPECT_FLOAT_EQ(out[25], 3);
    EXPECT_FLOAT_EQ(out[26], 0);
    EXPECT_FLOAT_EQ(out[27], 0);
    EXPECT_FLOAT_EQ(out[28], -7);
    EXPECT_FLOAT_EQ(out[29], -1);
    EXPECT_FLOAT_EQ(out[30], -2);
    EXPECT_FLOAT_EQ(out[31], -2);
    EXPECT_FLOAT_EQ(out[32], -2);
    EXPECT_FLOAT_EQ(out[33], 0);
    EXPECT_FLOAT_EQ(out[34], 0);
    EXPECT_FLOAT_EQ(out[35], 0);
    EXPECT_FLOAT_EQ(out[36], 4);
    EXPECT_FLOAT_EQ(out[37], 3);
    EXPECT_FLOAT_EQ(out[38], -2);
    EXPECT_FLOAT_EQ(out[39], -2);
    EXPECT_FLOAT_EQ(out[40], 4);
    EXPECT_FLOAT_EQ(out[41], 4);
    EXPECT_FLOAT_EQ(out[42], -2);
    EXPECT_FLOAT_EQ(out[43], 1);
    EXPECT_FLOAT_EQ(out[44], 1);
    EXPECT_FLOAT_EQ(out[45], 2);
    EXPECT_FLOAT_EQ(out[46], 2);
    EXPECT_FLOAT_EQ(out[47], 2);
}
