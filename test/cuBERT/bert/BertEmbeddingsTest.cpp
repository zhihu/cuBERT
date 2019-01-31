#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <cmath>

#include "cuBERT/common.h"
#include "cuBERT/bert/BertEmbeddings.h"
using namespace cuBERT;

class BertEmbeddingsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize();
        handle = cuBERT::blas_create();
    }

    void TearDown() override {
        cuBERT::blas_destroy(handle);
        cuBERT::finalize();
    }

    void* handle;
};

TEST_F(BertEmbeddingsTest, compute) {
    size_t vocab_size = 5;
    size_t type_vocab_size = 2;
    size_t hidden_size = 3;
    size_t seq_length = 4;

    float word_embeddings[15] = {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
            12, 13, 14,
    };
    float token_type_embeddings[6] = {
            -2, -1, 0,
            -5, -4, -3,
    };
    float position_embeddings[12] = {
            1, 1, 1,
            -1, -1, -1,
            2, 2, 2,
            -2, -2, -2,
    };

    float beta[3] = {1, 0, -1};
    float gamma[3] = {1, 1, 1};

    std::unordered_map<std::string, float *> var = {
            {"bert/embeddings/word_embeddings",       word_embeddings},
            {"bert/embeddings/token_type_embeddings", token_type_embeddings},
            {"bert/embeddings/position_embeddings",   position_embeddings},
            {"bert/embeddings/LayerNorm/beta", beta},
            {"bert/embeddings/LayerNorm/gamma", gamma},
    };

    BertEmbeddings bert_embeddings(handle, var, 32, vocab_size, type_vocab_size, hidden_size, seq_length);

    size_t batch_size = 2;
    int input_ids[8] = {3, 0, 1, 4,
                        2, 0, 2, 1};
    char segment_ids[8] = {1, 1, 0, 1,
                          0, 1, 0, 0};
    float out[24];

    int* input_ids_gpu;
    char* segment_ids_gpu;
    float* out_gpu;
    cudaMalloc(&input_ids_gpu, sizeof(int) * 8);
    cudaMalloc(&segment_ids_gpu, sizeof(char) * 8);
    cudaMalloc(&out_gpu, sizeof(float) * 24);
    cudaMemcpy(input_ids_gpu, input_ids, sizeof(int) * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(segment_ids_gpu, segment_ids, sizeof(char) * 8, cudaMemcpyHostToDevice);

    bert_embeddings.compute(batch_size, input_ids_gpu, segment_ids_gpu, out_gpu);

    cudaMemcpy(out, out_gpu, sizeof(float) * 24, cudaMemcpyDeviceToHost);
    cudaFree(input_ids_gpu);
    cudaFree(segment_ids_gpu);
    cudaFree(out_gpu);

    EXPECT_NEAR(out[0], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[1], 0);
    EXPECT_NEAR(out[2], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[3], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[4], 0);
    EXPECT_NEAR(out[5], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[6], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[7], 0);
    EXPECT_NEAR(out[8], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[9], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[10], 0);
    EXPECT_NEAR(out[11], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[12], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[13], 0);
    EXPECT_NEAR(out[14], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[15], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[16], 0);
    EXPECT_NEAR(out[17], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[18], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[19], 0);
    EXPECT_NEAR(out[20], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[21], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[22], 0);
    EXPECT_NEAR(out[23], std::sqrt(1.5) - 1, 1e-6);
}


class BertEmbeddingsCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize(true);
    }

    void TearDown() override {
        cuBERT::finalize();
    }
};

TEST_F(BertEmbeddingsCPUTest, compute_cpu) {
    size_t vocab_size = 5;
    size_t type_vocab_size = 2;
    size_t hidden_size = 3;
    size_t seq_length = 4;

    float word_embeddings[15] = {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
            12, 13, 14,
    };
    float token_type_embeddings[6] = {
            -2, -1, 0,
            -5, -4, -3,
    };
    float position_embeddings[12] = {
            1, 1, 1,
            -1, -1, -1,
            2, 2, 2,
            -2, -2, -2,
    };

    float beta[3] = {1, 0, -1};
    float gamma[3] = {1, 1, 1};

    std::unordered_map<std::string, float *> var = {
            {"bert/embeddings/word_embeddings",       word_embeddings},
            {"bert/embeddings/token_type_embeddings", token_type_embeddings},
            {"bert/embeddings/position_embeddings",   position_embeddings},
            {"bert/embeddings/LayerNorm/beta", beta},
            {"bert/embeddings/LayerNorm/gamma", gamma},
    };

    BertEmbeddings bert_embeddings(nullptr, var, 32, vocab_size, type_vocab_size, hidden_size, seq_length);

    size_t batch_size = 2;
    int input_ids[8] = {3, 0, 1, 4,
                        2, 0, 2, 1};
    char segment_ids[8] = {1, 1, 0, 1,
                           0, 1, 0, 0};
    float out[24];

    bert_embeddings.compute(batch_size, input_ids, segment_ids, out);

    EXPECT_NEAR(out[0], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[1], 0);
    EXPECT_NEAR(out[2], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[3], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[4], 0);
    EXPECT_NEAR(out[5], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[6], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[7], 0);
    EXPECT_NEAR(out[8], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[9], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[10], 0);
    EXPECT_NEAR(out[11], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[12], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[13], 0);
    EXPECT_NEAR(out[14], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[15], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[16], 0);
    EXPECT_NEAR(out[17], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[18], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[19], 0);
    EXPECT_NEAR(out[20], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[21], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[22], 0);
    EXPECT_NEAR(out[23], std::sqrt(1.5) - 1, 1e-6);
}
