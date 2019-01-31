#include "gtest/gtest.h"

#include "cuBERT/common.h"

class CommonTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CommonTest, compute) {
    cuBERT::initialize();

    float *array = (float *) cuBERT::malloc(sizeof(float) * 10);
    float *array_b = (float *) cuBERT::malloc(sizeof(float) * 10);

    cuBERT::memcpy(array, array_b, sizeof(float) * 10, 3);

    cuBERT::free(array_b);
    cuBERT::free(array);

    cuBERT::finalize();
}
