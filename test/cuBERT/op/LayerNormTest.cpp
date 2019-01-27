//
// Created by 田露 on 2019/1/17.
//

#include "gtest/gtest.h"
#include <cmath>

#include "cuBERT/op/LayerNorm.h"
using namespace cuBERT;

class LayerNormTest : public ::testing::Test {

};

TEST_F(LayerNormTest, compute_) {
    float beta[3] = {-1, 0, 1};
    float gamma[3] = {1, 2, 3};

    size_t max_batch_size = 2;
    LayerNorm layer_norm(max_batch_size, 3, beta, gamma);

    float inout[6] = {9, 10, 11, 5, 4, 3};

    float* inout_gpu;
    cudaMalloc(&inout_gpu, 6 * sizeof(float));
    cudaMemcpy(inout_gpu, inout, 6 * sizeof(float), cudaMemcpyHostToDevice);

    layer_norm.compute_(2, inout_gpu, nullptr);

    cudaMemcpy(inout, inout_gpu, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(inout_gpu);

    EXPECT_NEAR(inout[0], -2.224744871391589, 1e-5);
    EXPECT_FLOAT_EQ(inout[1], 0);
    EXPECT_NEAR(inout[2], 4.674234614174766, 1e-5);
    EXPECT_NEAR(inout[3], 0.22474487139158894, 1e-5);
    EXPECT_FLOAT_EQ(inout[4], 0);
    EXPECT_NEAR(inout[5], -2.674234614174767, 1e-5);
}

TEST_F(LayerNormTest, compute_cpu_) {
    float beta[3] = {-1, 0, 1};
    float gamma[3] = {1, 2, 3};

    size_t max_batch_size = 2;
    LayerNorm layer_norm(max_batch_size, 3, beta, gamma);

    float inout[6] = {9, 10, 11, 5, 4, 3};
    layer_norm.compute_cpu_(2, inout, nullptr);

    EXPECT_NEAR(inout[0], -2.224744871391589, 1e-5);
    EXPECT_FLOAT_EQ(inout[1], 0);
    EXPECT_NEAR(inout[2], 4.674234614174766, 1e-5);
    EXPECT_NEAR(inout[3], 0.22474487139158894, 1e-5);
    EXPECT_FLOAT_EQ(inout[4], 0);
    EXPECT_NEAR(inout[5], -2.674234614174767, 1e-5);
}

TEST_F(LayerNormTest, compute_cpu_ext_) {
    float beta[3] = {-1, 0, 1};
    float gamma[3] = {1, 2, 3};

    size_t max_batch_size = 2;
    LayerNorm layer_norm(max_batch_size, 3, beta, gamma);

    float in[6] = {8, 8, 8, 2, 2, 2};
    float inout[6] = {1, 2, 3, 3, 2, 1};
    layer_norm.compute_cpu_(2, in, inout, nullptr);

    EXPECT_NEAR(inout[0], -2.224744871391589, 1e-5);
    EXPECT_FLOAT_EQ(inout[1], 0);
    EXPECT_NEAR(inout[2], 4.674234614174766, 1e-5);
    EXPECT_NEAR(inout[3], 0.22474487139158894, 1e-5);
    EXPECT_FLOAT_EQ(inout[4], 0);
    EXPECT_NEAR(inout[5], -2.674234614174767, 1e-5);
}
