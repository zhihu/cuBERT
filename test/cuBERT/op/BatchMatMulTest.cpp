#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuBERT/op/BatchMatMul.h"
using namespace cuBERT;

class BatchMatMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        cublasCreate_v2(&handle);
    }

    void TearDown() override {
        cublasDestroy_v2(handle);
    }

    cublasHandle_t handle;
};

TEST_F(BatchMatMulTest, compute) {
    float A[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    float B[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    float out[24];

    float* A_gpu;
    float* B_gpu;
    float* out_gpu;

    cudaMalloc(&A_gpu, 12 * sizeof(float));
    cudaMalloc(&B_gpu, 16 * sizeof(float));
    cudaMalloc(&out_gpu, 24 * sizeof(float));

    cudaMemcpy(A_gpu, A, 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, 16 * sizeof(float), cudaMemcpyHostToDevice);

    BatchMatMul batch_matmul(handle, false, false, 3, 4, 2, 32);

    batch_matmul.compute(2, A_gpu, B_gpu, out_gpu);

    cudaMemcpy(out, out_gpu, 24 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(out_gpu);
    cudaFree(B_gpu);
    cudaFree(A_gpu);

    EXPECT_FLOAT_EQ(out[0], 11);
    EXPECT_FLOAT_EQ(out[1], 10);
    EXPECT_FLOAT_EQ(out[2], 9);
    EXPECT_FLOAT_EQ(out[3], 8);
    EXPECT_FLOAT_EQ(out[4], 63);
    EXPECT_FLOAT_EQ(out[5], 58);
    EXPECT_FLOAT_EQ(out[6], 53);
    EXPECT_FLOAT_EQ(out[7], 48);
    EXPECT_FLOAT_EQ(out[8], 115);
    EXPECT_FLOAT_EQ(out[9], 106);
    EXPECT_FLOAT_EQ(out[10], 97);
    EXPECT_FLOAT_EQ(out[11], 88);
    EXPECT_FLOAT_EQ(out[12], 63);
    EXPECT_FLOAT_EQ(out[13], 50);
    EXPECT_FLOAT_EQ(out[14], 37);
    EXPECT_FLOAT_EQ(out[15], 24);
    EXPECT_FLOAT_EQ(out[16], 83);
    EXPECT_FLOAT_EQ(out[17], 66);
    EXPECT_FLOAT_EQ(out[18], 49);
    EXPECT_FLOAT_EQ(out[19], 32);
    EXPECT_FLOAT_EQ(out[20], 103);
    EXPECT_FLOAT_EQ(out[21], 82);
    EXPECT_FLOAT_EQ(out[22], 61);
    EXPECT_FLOAT_EQ(out[23], 40);
}

TEST_F(BatchMatMulTest, compute_t) {
    float A[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    float B[12] = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    float out[18];

    float* A_gpu;
    float* B_gpu;
    float* out_gpu;

    cudaMalloc(&A_gpu, 12 * sizeof(float));
    cudaMalloc(&B_gpu, 12 * sizeof(float));
    cudaMalloc(&out_gpu, 18 * sizeof(float));

    cudaMemcpy(A_gpu, A, 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, 12 * sizeof(float), cudaMemcpyHostToDevice);

    BatchMatMul batch_matmul(handle, false, true, 3, 3, 2, 32);

    batch_matmul.compute(2, A_gpu, B_gpu, out_gpu);

    cudaMemcpy(out, out_gpu, 18 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(out_gpu);
    cudaFree(B_gpu);
    cudaFree(A_gpu);

    EXPECT_FLOAT_EQ(out[0], 10);
    EXPECT_FLOAT_EQ(out[1], 8);
    EXPECT_FLOAT_EQ(out[2], 6);
    EXPECT_FLOAT_EQ(out[3], 52);
    EXPECT_FLOAT_EQ(out[4], 42);
    EXPECT_FLOAT_EQ(out[5], 32);
    EXPECT_FLOAT_EQ(out[6], 94);
    EXPECT_FLOAT_EQ(out[7], 76);
    EXPECT_FLOAT_EQ(out[8], 58);
    EXPECT_FLOAT_EQ(out[9], 58);
    EXPECT_FLOAT_EQ(out[10], 32);
    EXPECT_FLOAT_EQ(out[11], 6);
    EXPECT_FLOAT_EQ(out[12], 76);
    EXPECT_FLOAT_EQ(out[13], 42);
    EXPECT_FLOAT_EQ(out[14], 8);
    EXPECT_FLOAT_EQ(out[15], 94);
    EXPECT_FLOAT_EQ(out[16], 52);
    EXPECT_FLOAT_EQ(out[17], 10);
}
