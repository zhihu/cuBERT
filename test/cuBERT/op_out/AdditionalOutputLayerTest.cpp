#include "gtest/gtest.h"

#include "../common_test.h"
#include "cuBERT/op_out/AdditionalOutputLayer.h"
using namespace cuBERT;


TEST_F(CommonTest, additional_output) {
    size_t hidden_size = 3;
    float output_weights[3] = {-1, 0, 1};

    ClassifierOutputLayer<float> aol(handle, hidden_size, 1, output_weights, nullptr, 4);

    float in[6] = {
            2, 8, 3,
            -4, 1, 2,
    };
    float out[2];

    float* in_gpu = (float*) cuBERT::malloc(sizeof(float) * 6);
    float* out_gpu = (float*) cuBERT::malloc(sizeof(float) * 2);

    cuBERT::memcpy(in_gpu, in, sizeof(float) * 6, 1);

    aol.compute(2, in_gpu, out_gpu);

    cuBERT::memcpy(out, out_gpu, sizeof(float) * 2, 2);
    cuBERT::free(in_gpu);
    cuBERT::free(out_gpu);

    EXPECT_FLOAT_EQ(out[0], 1);
    EXPECT_FLOAT_EQ(out[1], 6);
}
