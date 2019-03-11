#include "gtest/gtest.h"

#include "../common_test.h"
#include "cuBERT/op/Softmax.h"
using namespace cuBERT;

TEST_F(CommonTest, softmax_) {
    float inout[6] = {1, 2, 3, 6, 6, 6};
    float *inout_gpu = (float*) cuBERT::malloc(sizeof(float) * 6);
    cuBERT::memcpy(inout_gpu, inout, sizeof(float) * 6, 1);

    Softmax softmax(4, 3);
    softmax.compute_(2, inout_gpu, nullptr);

    cuBERT::memcpy(inout, inout_gpu, sizeof(float) * 6, 2);
    cuBERT::free(inout_gpu);

    EXPECT_FLOAT_EQ(inout[0], 0.090030573);
    EXPECT_FLOAT_EQ(inout[1], 0.244728478);
    EXPECT_FLOAT_EQ(inout[2], 0.66524094);
    EXPECT_FLOAT_EQ(inout[3], 0.33333334);
    EXPECT_FLOAT_EQ(inout[4], 0.33333334);
    EXPECT_FLOAT_EQ(inout[5], 0.33333334);
}
