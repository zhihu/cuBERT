#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuBERT/common.h"
#include "cuBERT/op/Dense.h"
using namespace cuBERT;

class DenseTest : public ::testing::Test {
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

TEST_F(DenseTest, compute) {
    float kernel[6] = {1, 2, 3, 4, 5, 6};
    float bias[3] = {-3, -2, -1};

    Dense dense(handle, 2, 3, kernel, bias, 16);

    float input[6] = {1, 2, 3, 4, 5, 6};
    float *input_gpu;
    cudaMalloc(&input_gpu, 6 * sizeof(float));
    cublasSetMatrix(2, 3, sizeof(float), input, 2, input_gpu, 2);

    float *output_gpu;
    cudaMalloc(&output_gpu, 9 * sizeof(float));

    dense.compute(3, input_gpu, output_gpu);

    float output[9];
    cublasGetMatrix(3, 3, sizeof(float), output_gpu, 3, output, 3);

    cudaFree(output_gpu);
    cudaFree(input_gpu);

    EXPECT_FLOAT_EQ(output[0], 9 - 3);
    EXPECT_FLOAT_EQ(output[1], 12 - 2);
    EXPECT_FLOAT_EQ(output[2], 15 - 1);
    EXPECT_FLOAT_EQ(output[3], 19 - 3);
    EXPECT_FLOAT_EQ(output[4], 26 - 2);
    EXPECT_FLOAT_EQ(output[5], 33 - 1);
    EXPECT_FLOAT_EQ(output[6], 29 - 3);
    EXPECT_FLOAT_EQ(output[7], 40 - 2);
    EXPECT_FLOAT_EQ(output[8], 51 - 1);
}

class DenseCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize(true);
    }

    void TearDown() override {
        cuBERT::finalize();
    }
};

TEST_F(DenseCPUTest, compute) {
    float kernel[6] = {1, 2, 3, 4, 5, 6};
    float bias[3] = {-3, -2, -1};

    Dense dense(nullptr, 2, 3, kernel, bias, 16);

    float input[6] = {1, 2, 3, 4, 5, 6};
    float output[9];
    dense.compute(3, input, output);

    EXPECT_FLOAT_EQ(output[0], 9 - 3);
    EXPECT_FLOAT_EQ(output[1], 12 - 2);
    EXPECT_FLOAT_EQ(output[2], 15 - 1);
    EXPECT_FLOAT_EQ(output[3], 19 - 3);
    EXPECT_FLOAT_EQ(output[4], 26 - 2);
    EXPECT_FLOAT_EQ(output[5], 33 - 1);
    EXPECT_FLOAT_EQ(output[6], 29 - 3);
    EXPECT_FLOAT_EQ(output[7], 40 - 2);
    EXPECT_FLOAT_EQ(output[8], 51 - 1);
}
