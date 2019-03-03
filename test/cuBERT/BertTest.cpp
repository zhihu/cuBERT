#include "gtest/gtest.h"
#include <unordered_map>

#include "common_test.h"
#include "cuBERT/Bert.h"
#include "cuBERT/tensorflow/Graph.h"
using namespace cuBERT;

class BertTest : public CommonTest {
protected:
    void SetUp() override {
        CommonTest::SetUp();
        graph = new Graph("bert_frozen_seq32.pb");
    }

    void TearDown() override {
        delete graph;
        CommonTest::TearDown();
    }

    Graph* graph;
};

TEST_F(BertTest, compute) {
    Bert bert(graph->var, 128, 32, graph->vocab_size, graph->type_vocab_size);

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

    bert.compute(2, input_ids, input_mask, segment_ids);

    float embedding_output[49152];
    bert.embedding_output(2, embedding_output);
    EXPECT_FLOAT_EQ(embedding_output[0], 0.1593448);
    EXPECT_FLOAT_EQ(embedding_output[1], 0.21887021);
    EXPECT_FLOAT_EQ(embedding_output[2], -0.3861023);
    EXPECT_FLOAT_EQ(embedding_output[49149], 2.4485614);
    EXPECT_FLOAT_EQ(embedding_output[49150], -0.029199962);
    EXPECT_FLOAT_EQ(embedding_output[49151], 0.33240327);

    float logits[2];
    bert.logits(2, logits);
    EXPECT_NEAR(logits[0], -2.9427543, 1e-5);
    EXPECT_NEAR(logits[1], -1.4876306, 1e-5);
}
