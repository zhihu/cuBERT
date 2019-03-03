#include "gtest/gtest.h"

#include "common_test.h"

TEST_F(CommonTest, memcpy) {
    float *array = (float *) cuBERT::malloc(sizeof(float) * 10);
    float *array_b = (float *) cuBERT::malloc(sizeof(float) * 10);

    cuBERT::memcpy(array, array_b, sizeof(float) * 10, 3);

    cuBERT::free(array_b);
    cuBERT::free(array);
}
