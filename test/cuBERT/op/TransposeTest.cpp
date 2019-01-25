//
// Created by 田露 on 2019/1/17.
//

#include "gtest/gtest.h"
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "cuBERT/op/Transpose.h"
using namespace cuBERT;

class TransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudnnCreate(&handle);
    }

    void TearDown() override {
        cudnnDestroy(handle);
    }

    cudnnHandle_t handle;
};

TEST_F(TransposeTest, compute) {
    std::vector<int> dims_in{-1, 3, 1, 5};
    std::vector<int> axes{0, 3, 1, 2};
    Transpose transpose(handle, dims_in, axes);

    float in[30] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    float out[30];

    float* in_gpu;
    float* out_gpu;
    cudaMalloc(&in_gpu, 30 * sizeof(float));
    cudaMalloc(&out_gpu, 30 * sizeof(float));

    cudaMemcpy(in_gpu, in, 30 * sizeof(float), cudaMemcpyHostToDevice);

    transpose.compute(2, in_gpu, out_gpu);

    cudaMemcpy(out, out_gpu, 30 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(out_gpu);
    cudaFree(in_gpu);

    EXPECT_FLOAT_EQ(out[0], 0);
    EXPECT_FLOAT_EQ(out[1], 5);
    EXPECT_FLOAT_EQ(out[2], 10);
    EXPECT_FLOAT_EQ(out[3], 1);
    EXPECT_FLOAT_EQ(out[4], 6);
    EXPECT_FLOAT_EQ(out[5], 11);
    EXPECT_FLOAT_EQ(out[6], 2);
    EXPECT_FLOAT_EQ(out[7], 7);
    EXPECT_FLOAT_EQ(out[8], 12);
    EXPECT_FLOAT_EQ(out[9], 3);
    EXPECT_FLOAT_EQ(out[10], 8);
    EXPECT_FLOAT_EQ(out[11], 13);
    EXPECT_FLOAT_EQ(out[12], 4);
    EXPECT_FLOAT_EQ(out[13], 9);
    EXPECT_FLOAT_EQ(out[14], 14);
    EXPECT_FLOAT_EQ(out[15], 15);
    EXPECT_FLOAT_EQ(out[16], 20);
    EXPECT_FLOAT_EQ(out[17], 25);
    EXPECT_FLOAT_EQ(out[18], 16);
    EXPECT_FLOAT_EQ(out[19], 21);
    EXPECT_FLOAT_EQ(out[20], 26);
    EXPECT_FLOAT_EQ(out[21], 17);
    EXPECT_FLOAT_EQ(out[22], 22);
    EXPECT_FLOAT_EQ(out[23], 27);
    EXPECT_FLOAT_EQ(out[24], 18);
    EXPECT_FLOAT_EQ(out[25], 23);
    EXPECT_FLOAT_EQ(out[26], 28);
    EXPECT_FLOAT_EQ(out[27], 19);
    EXPECT_FLOAT_EQ(out[28], 24);
    EXPECT_FLOAT_EQ(out[29], 29);
}

TEST_F(TransposeTest, compute_cpu) {
    std::vector<int> dims_in{-1, 3, 1, 5};
    std::vector<int> axes{0, 3, 1, 2};
    Transpose transpose(handle, dims_in, axes);

    float in[30] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    float out[30];
    transpose.compute_cpu(2, in, out);

    EXPECT_FLOAT_EQ(out[0], 0);
    EXPECT_FLOAT_EQ(out[1], 5);
    EXPECT_FLOAT_EQ(out[2], 10);
    EXPECT_FLOAT_EQ(out[3], 1);
    EXPECT_FLOAT_EQ(out[4], 6);
    EXPECT_FLOAT_EQ(out[5], 11);
    EXPECT_FLOAT_EQ(out[6], 2);
    EXPECT_FLOAT_EQ(out[7], 7);
    EXPECT_FLOAT_EQ(out[8], 12);
    EXPECT_FLOAT_EQ(out[9], 3);
    EXPECT_FLOAT_EQ(out[10], 8);
    EXPECT_FLOAT_EQ(out[11], 13);
    EXPECT_FLOAT_EQ(out[12], 4);
    EXPECT_FLOAT_EQ(out[13], 9);
    EXPECT_FLOAT_EQ(out[14], 14);
    EXPECT_FLOAT_EQ(out[15], 15);
    EXPECT_FLOAT_EQ(out[16], 20);
    EXPECT_FLOAT_EQ(out[17], 25);
    EXPECT_FLOAT_EQ(out[18], 16);
    EXPECT_FLOAT_EQ(out[19], 21);
    EXPECT_FLOAT_EQ(out[20], 26);
    EXPECT_FLOAT_EQ(out[21], 17);
    EXPECT_FLOAT_EQ(out[22], 22);
    EXPECT_FLOAT_EQ(out[23], 27);
    EXPECT_FLOAT_EQ(out[24], 18);
    EXPECT_FLOAT_EQ(out[25], 23);
    EXPECT_FLOAT_EQ(out[26], 28);
    EXPECT_FLOAT_EQ(out[27], 19);
    EXPECT_FLOAT_EQ(out[28], 24);
    EXPECT_FLOAT_EQ(out[29], 29);
}
