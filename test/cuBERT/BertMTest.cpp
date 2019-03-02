#include "gtest/gtest.h"

#include "cuBERT/common.h"
#include "cuBERT/BertM.h"
using namespace cuBERT;

class BertMTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize();
    }

    void TearDown() override {
        cuBERT::finalize();
    }
};

TEST_F(BertMTest, compute) {
    BertM bert("bert_frozen_seq32.pb", 128, 32);

    int input_ids[] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    };

    char input_mask[] = {
            1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0};

    char segment_ids[] = {
            1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
            0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0};

    float logits[2];

    bert.compute(2, input_ids, input_mask, segment_ids, logits);
    EXPECT_FLOAT_EQ(logits[0], -2.9427543);
    EXPECT_FLOAT_EQ(logits[1], -1.4876306);

    bert.compute(2, input_ids, input_mask, segment_ids, logits);
    EXPECT_FLOAT_EQ(logits[0], -2.9427543);
    EXPECT_FLOAT_EQ(logits[1], -1.4876306);

    bert.compute(2, input_ids, input_mask, segment_ids, logits);
    EXPECT_FLOAT_EQ(logits[0], -2.9427543);
    EXPECT_FLOAT_EQ(logits[1], -1.4876306);

    bert.compute(2, input_ids, input_mask, segment_ids, logits);
    EXPECT_FLOAT_EQ(logits[0], -2.9427543);
    EXPECT_FLOAT_EQ(logits[1], -1.4876306);
}
