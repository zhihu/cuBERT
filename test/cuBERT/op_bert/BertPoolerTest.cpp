#include "gtest/gtest.h"
#include <cmath>

#include "../common_test.h"
#include "cuBERT/op_bert/BertPooler.h"
using namespace cuBERT;

TEST_F(CommonTest, bert_pooler) {
    size_t seq_length = 3;
    size_t hidden_size = 2;

    float kernel[] = {-1, 0,
                      0, 1};
    float bias[] = {2, 3};

    BertPooler pooler(handle, seq_length, hidden_size, kernel, bias, 32);

    float in[12] = {
            0, 1,
            1, 1,
            2, 1,
            -2, -1,
            3, 2,
            0, 5,
    };
    float out[4];

    float* in_gpu = (float*) cuBERT::malloc(sizeof(float) * 12);
    float* out_gpu = (float*) cuBERT::malloc(sizeof(float) * 4);

    cuBERT::memcpy(in_gpu, in, sizeof(float) * 12, 1);

    pooler.compute(2, in_gpu, out_gpu);

    cuBERT::memcpy(out, out_gpu, sizeof(float) * 4, 2);
    cuBERT::free(in_gpu);
    cuBERT::free(out_gpu);

    EXPECT_FLOAT_EQ(out[0], tanhf(2));
    EXPECT_FLOAT_EQ(out[1], tanhf(4));
    EXPECT_FLOAT_EQ(out[2], tanhf(4));
    EXPECT_FLOAT_EQ(out[3], tanhf(2));
}
