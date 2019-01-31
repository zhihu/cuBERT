#include "gtest/gtest.h"

#include "cuBERT/common.h"
#include "cuBERT/op/Softmax.h"
using namespace cuBERT;

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize();
    }

    void TearDown() override {
        cuBERT::finalize();
    }
};

TEST_F(SoftmaxTest, compute_) {
    float inout[6] = {1, 2, 3, 6, 6, 6};
    float *inout_gpu;

    cudaMalloc(&inout_gpu, sizeof(float) * 6);
    cudaMemcpy(inout_gpu, inout, sizeof(float) * 6, cudaMemcpyHostToDevice);

    Softmax softmax(4, 3);
    softmax.compute_(2, inout_gpu, nullptr);

    cudaMemcpy(inout, inout_gpu, sizeof(float) * 6, cudaMemcpyDeviceToHost);
    cudaFree(inout_gpu);

    EXPECT_FLOAT_EQ(inout[0], 0.090030573);
    EXPECT_FLOAT_EQ(inout[1], 0.244728478);
    EXPECT_FLOAT_EQ(inout[2], 0.66524094);
    EXPECT_FLOAT_EQ(inout[3], 0.33333334);
    EXPECT_FLOAT_EQ(inout[4], 0.33333334);
    EXPECT_FLOAT_EQ(inout[5], 0.33333334);
}


class SoftmaxCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize(true);
    }

    void TearDown() override {
        cuBERT::finalize();
    }
};

TEST_F(SoftmaxCPUTest, compute_cpu_) {
    float inout[6] = {1, 2, 3, 6, 6, 6};

    Softmax softmax(4, 3);
    softmax.compute_(2, inout, nullptr);

    EXPECT_FLOAT_EQ(inout[0], 0.090030573);
    EXPECT_FLOAT_EQ(inout[1], 0.244728478);
    EXPECT_FLOAT_EQ(inout[2], 0.66524094);
    EXPECT_FLOAT_EQ(inout[3], 0.33333334);
    EXPECT_FLOAT_EQ(inout[4], 0.33333334);
    EXPECT_FLOAT_EQ(inout[5], 0.33333334);
}
