#include "gtest/gtest.h"
#include <cmath>

#include "../common_test.h"
#include "cuBERT/op/GELU.h"
using namespace cuBERT;

TEST_F(CommonTest, gelu) {
    float inout[5] = {-2, -1, 0, 1, 2};

    float expect[5];
    for (int i = 0; i < 5; ++i) {
        expect[i] = inout[i] * 0.5 * (1.0 + erf(inout[i] / sqrt(2.0)));
    }

    GELU gelu;

    float* inout_gpu = (float*) cuBERT::malloc(sizeof(float) * 5);
    cuBERT::memcpy(inout_gpu, inout, sizeof(float) * 5, 1);

    gelu.compute_(5, inout_gpu, nullptr);

    cuBERT::memcpy(inout, inout_gpu, sizeof(float) * 5, 2);
    cuBERT::free(inout_gpu);

    EXPECT_NEAR(inout[0], expect[0], 1e-5);
    EXPECT_NEAR(inout[1], expect[1], 1e-5);
    EXPECT_NEAR(inout[2], expect[2], 1e-5);
    EXPECT_NEAR(inout[3], expect[3], 1e-5);
    EXPECT_NEAR(inout[4], expect[4], 1e-5);
    EXPECT_NEAR(inout[5], expect[5], 1e-5);
}
