#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuBERT/common.h"
#include "cuBERT/bert/AttentionMask.h"
using namespace cuBERT;

class AttentionMaskTest : public ::testing::Test {
protected:
    void SetUp() override {
        cublasCreate_v2(&handle);
        cuBERT::initialize();
    }

    void TearDown() override {
        cublasDestroy_v2(handle);
        cuBERT::finalize();
    }

    cublasHandle_t handle;
};

TEST_F(AttentionMaskTest, compute) {
    size_t seq_length = 2;
    size_t num_attention_heads = 3;
    size_t batch_size = 2;

    AttentionMask attention_mask(handle, seq_length, num_attention_heads, 4);

    char in[] = {
            1, 0,
            0, 0,
    };
    float out[24];

    char* in_gpu;
    float* out_gpu;
    cudaMalloc(&in_gpu, sizeof(char) * 4);
    cudaMalloc(&out_gpu, sizeof(float) * 24);

    cudaMemcpy(in_gpu, in, sizeof(char) * 4, cudaMemcpyHostToDevice);

    attention_mask.compute(batch_size, in_gpu, out_gpu);

    cudaMemcpy(out, out_gpu, sizeof(float) * 24, cudaMemcpyDeviceToHost);
    cudaFree(in_gpu);
    cudaFree(out_gpu);

    EXPECT_FLOAT_EQ(out[0], 0);
    EXPECT_FLOAT_EQ(out[1], 1);
    EXPECT_FLOAT_EQ(out[2], 0);
    EXPECT_FLOAT_EQ(out[3], 1);
    EXPECT_FLOAT_EQ(out[4], 0);
    EXPECT_FLOAT_EQ(out[5], 1);
    EXPECT_FLOAT_EQ(out[6], 0);
    EXPECT_FLOAT_EQ(out[7], 1);
    EXPECT_FLOAT_EQ(out[8], 0);
    EXPECT_FLOAT_EQ(out[9], 1);
    EXPECT_FLOAT_EQ(out[10], 0);
    EXPECT_FLOAT_EQ(out[11], 1);
    EXPECT_FLOAT_EQ(out[12], 1);
    EXPECT_FLOAT_EQ(out[13], 1);
    EXPECT_FLOAT_EQ(out[14], 1);
    EXPECT_FLOAT_EQ(out[15], 1);
    EXPECT_FLOAT_EQ(out[16], 1);
    EXPECT_FLOAT_EQ(out[17], 1);
    EXPECT_FLOAT_EQ(out[18], 1);
    EXPECT_FLOAT_EQ(out[19], 1);
    EXPECT_FLOAT_EQ(out[20], 1);
    EXPECT_FLOAT_EQ(out[21], 1);
    EXPECT_FLOAT_EQ(out[22], 1);
    EXPECT_FLOAT_EQ(out[23], 1);
}


class AttentionMaskCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize(true);
    }

    void TearDown() override {
        cuBERT::finalize();
    }
};

TEST_F(AttentionMaskCPUTest, compute_cpu) {
    size_t seq_length = 2;
    size_t num_attention_heads = 3;
    size_t batch_size = 2;

    AttentionMask attention_mask(nullptr, seq_length, num_attention_heads, 4);

    char in[] = {
            1, 0,
            0, 0,
    };
    float out[24];

    attention_mask.compute(batch_size, in, out);

    EXPECT_FLOAT_EQ(out[0], 0);
    EXPECT_FLOAT_EQ(out[1], 1);
    EXPECT_FLOAT_EQ(out[2], 0);
    EXPECT_FLOAT_EQ(out[3], 1);
    EXPECT_FLOAT_EQ(out[4], 0);
    EXPECT_FLOAT_EQ(out[5], 1);
    EXPECT_FLOAT_EQ(out[6], 0);
    EXPECT_FLOAT_EQ(out[7], 1);
    EXPECT_FLOAT_EQ(out[8], 0);
    EXPECT_FLOAT_EQ(out[9], 1);
    EXPECT_FLOAT_EQ(out[10], 0);
    EXPECT_FLOAT_EQ(out[11], 1);
    EXPECT_FLOAT_EQ(out[12], 1);
    EXPECT_FLOAT_EQ(out[13], 1);
    EXPECT_FLOAT_EQ(out[14], 1);
    EXPECT_FLOAT_EQ(out[15], 1);
    EXPECT_FLOAT_EQ(out[16], 1);
    EXPECT_FLOAT_EQ(out[17], 1);
    EXPECT_FLOAT_EQ(out[18], 1);
    EXPECT_FLOAT_EQ(out[19], 1);
    EXPECT_FLOAT_EQ(out[20], 1);
    EXPECT_FLOAT_EQ(out[21], 1);
    EXPECT_FLOAT_EQ(out[22], 1);
    EXPECT_FLOAT_EQ(out[23], 1);
}
