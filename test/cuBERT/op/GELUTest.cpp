//
// Created by 田露 on 2019/1/21.
//

#include "gtest/gtest.h"
#include <cmath>

#include "cuBERT/op/GELU.h"
using namespace cuBERT;

class GELUTest : public ::testing::Test {

};

TEST_F(GELUTest, compute_cpu_) {
    float inout[5] = {-2, -1, 0, 1, 2};

    float expect[5];
    for (int i = 0; i < 5; ++i) {
        expect[i] = inout[i] * 0.5 * (1.0 + erf(inout[i] / sqrt(2.0)));
    }

    GELU gelu;
    gelu.compute_cpu_(5, inout, nullptr);

    EXPECT_NEAR(inout[0], expect[0], 1e-5);
    EXPECT_NEAR(inout[1], expect[1], 1e-5);
    EXPECT_NEAR(inout[2], expect[2], 1e-5);
    EXPECT_NEAR(inout[3], expect[3], 1e-5);
    EXPECT_NEAR(inout[4], expect[4], 1e-5);
    EXPECT_NEAR(inout[5], expect[5], 1e-5);
}
