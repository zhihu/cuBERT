#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

#include "cuBERT/bert/AttentionSelf.h"
using namespace cuBERT;

class AttentionSelfTest : public ::testing::Test {
protected:
    void SetUp() override {
        cublasCreate_v2(&cublas);
    }

    void TearDown() override {
        cublasDestroy_v2(cublas);
    }

    cublasHandle_t cublas;
};

TEST_F(AttentionSelfTest, compute) {
    float query_kernel[] = {-0.07848196, -0.18097023, 0.06933199, -0.07760319, 0.11389876, 0.05236414,
                            -0.02015782, 0.00233333, -0.00281469, -0.01525305, 0.17362033, -0.01600084,
                            0.00521428, 0.06063714, -0.10533229, 0.0228875, -0.00108843, -0.05974746,
                            -0.05530503, 0.06056419, 0.099603, 0.04929306, 0.08636444, 0.08424559,
                            0.02739674, -0.08676406, -0.0819858, 0.03834791, -0.03903558, 0.01903536,
                            0.01325864, 0.07587593, 0.20709228, -0.0421985, -0.10500058, -0.08004139};
    float query_bias[] = {-0.01566293, -0.01429354, -0.02946532, 0.02332242, -0.03551506, 0.00519018};

    float key_kernel[] = {-0.19046976, -0.052291, 0.00774184, -0.04793982, -0.03272828, -0.07022775,
                          0.05397043, 0.22157724, -0.28796428, -0.13628182, 0.10769557, -0.04396444,
                          0.11023977, 0.11277004, -0.17019109, -0.00998783, -0.13503011, 0.03862515,
                          -0.00570178, -0.03683843, -0.09878516, -0.08536254, -0.20706373, 0.07736684,
                          0.09753255, 0.08549864, 0.07573727, -0.08459285, 0.11262332, -0.06660723,
                          -0.05978908, 0.04687774, 0.20048976, -0.15552515, -0.09287686, -0.05736409};
    float key_bias[] = {0.01119683, -0.00749641, 0.00929781, -0.00789247, 0.00374282, -0.0203852};

    float value_kernel[] = {0.18298741, 0.13052676, 0.13003705, -0.07762788, -0.11298412, -0.09672086,
                            -0.27567647, -0.11159269, -0.20191047, -0.04961415, 0.03338585, -0.00217377,
                            0.0080993, -0.0092568, -0.07923323, -0.09595821, -0.0724212, 0.00234286,
                            0.08350474, 0.10685625, -0.03265393, 0.12026393, 0.11865459, 0.03879681,
                            0.09247954, -0.08354547, -0.04044447, 0.05576184, 0.063286, -0.06426957,
                            0.11189654, 0.04743394, 0.04952021, 0.06824017, -0.0718908, 0.06118326};
    float value_bias[] = {-0.01532887, -0.02567805, 0.02993296, 0.00255634, 0.03075514, -0.02086536};

    std::unordered_map<std::string, float*> var = {
            {"attention/self/query/kernel", query_kernel},
            {"attention/self/query/bias", query_bias},
            {"attention/self/key/kernel", key_kernel},
            {"attention/self/key/bias", key_bias},
            {"attention/self/value/kernel", value_kernel},
            {"attention/self/value/bias", value_bias},
    };

    size_t batch_size = 2;
    size_t num_attention_heads = 2;
    size_t size_per_head = 3;
    size_t seq_length = 4;

    float tensor[48] = {0, 1, 2, 3, 4, 5,
                        6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47};

    float mask[64];
    std::fill_n(mask, 64, 0);
    mask[0] = 1;

    float out[48];

    // gpu
    float* tensor_gpu;
    float* mask_gpu;
    float* out_gpu;
    cudaMalloc(&tensor_gpu, sizeof(float) * 48);
    cudaMalloc(&mask_gpu, sizeof(float) * 64);
    cudaMalloc(&out_gpu, sizeof(float) * 48);
    cudaMemcpy(tensor_gpu, tensor, sizeof(float) * 48, cudaMemcpyHostToDevice);
    cudaMemcpy(mask_gpu, mask, sizeof(float) * 64, cudaMemcpyHostToDevice);

    // compute
    AttentionSelf attention_self(cublas, "attention/self", var, 32, seq_length,
                                 out_gpu,
                                 num_attention_heads * size_per_head, num_attention_heads, size_per_head);
    attention_self.compute(batch_size, tensor_gpu, mask_gpu);

    cudaMemcpy(out, out_gpu, sizeof(float) * 48, cudaMemcpyDeviceToHost);
    cudaFree(out_gpu);
    cudaFree(mask_gpu);
    cudaFree(tensor_gpu);

    EXPECT_FLOAT_EQ(out[0], 3.1048317);
    EXPECT_FLOAT_EQ(out[1], 0.9379833);
    EXPECT_FLOAT_EQ(out[2], -2.2327702);
    EXPECT_FLOAT_EQ(out[3], 0.8868696);
    EXPECT_FLOAT_EQ(out[4], -0.2311137);
    EXPECT_FLOAT_EQ(out[5], -0.43307996);
    EXPECT_FLOAT_EQ(out[42], 5.7841063);
    EXPECT_FLOAT_EQ(out[43], 1.9979048);
    EXPECT_FLOAT_EQ(out[44], -4.535029);
    EXPECT_FLOAT_EQ(out[45], 1.191874);
    EXPECT_FLOAT_EQ(out[46], -0.8387809);
    EXPECT_FLOAT_EQ(out[47], -1.3139831);
}

TEST_F(AttentionSelfTest, compute_cpu) {
    float query_kernel[] = {-0.07848196, -0.18097023, 0.06933199, -0.07760319, 0.11389876, 0.05236414,
                            -0.02015782, 0.00233333, -0.00281469, -0.01525305, 0.17362033, -0.01600084,
                            0.00521428, 0.06063714, -0.10533229, 0.0228875, -0.00108843, -0.05974746,
                            -0.05530503, 0.06056419, 0.099603, 0.04929306, 0.08636444, 0.08424559,
                            0.02739674, -0.08676406, -0.0819858, 0.03834791, -0.03903558, 0.01903536,
                            0.01325864, 0.07587593, 0.20709228, -0.0421985, -0.10500058, -0.08004139};
    float query_bias[] = {-0.01566293, -0.01429354, -0.02946532, 0.02332242, -0.03551506, 0.00519018};

    float key_kernel[] = {-0.19046976, -0.052291, 0.00774184, -0.04793982, -0.03272828, -0.07022775,
                          0.05397043, 0.22157724, -0.28796428, -0.13628182, 0.10769557, -0.04396444,
                          0.11023977, 0.11277004, -0.17019109, -0.00998783, -0.13503011, 0.03862515,
                          -0.00570178, -0.03683843, -0.09878516, -0.08536254, -0.20706373, 0.07736684,
                          0.09753255, 0.08549864, 0.07573727, -0.08459285, 0.11262332, -0.06660723,
                          -0.05978908, 0.04687774, 0.20048976, -0.15552515, -0.09287686, -0.05736409};
    float key_bias[] = {0.01119683, -0.00749641, 0.00929781, -0.00789247, 0.00374282, -0.0203852};

    float value_kernel[] = {0.18298741, 0.13052676, 0.13003705, -0.07762788, -0.11298412, -0.09672086,
                            -0.27567647, -0.11159269, -0.20191047, -0.04961415, 0.03338585, -0.00217377,
                            0.0080993, -0.0092568, -0.07923323, -0.09595821, -0.0724212, 0.00234286,
                            0.08350474, 0.10685625, -0.03265393, 0.12026393, 0.11865459, 0.03879681,
                            0.09247954, -0.08354547, -0.04044447, 0.05576184, 0.063286, -0.06426957,
                            0.11189654, 0.04743394, 0.04952021, 0.06824017, -0.0718908, 0.06118326};
    float value_bias[] = {-0.01532887, -0.02567805, 0.02993296, 0.00255634, 0.03075514, -0.02086536};

    std::unordered_map<std::string, float*> var = {
            {"attention/self/query/kernel", query_kernel},
            {"attention/self/query/bias", query_bias},
            {"attention/self/key/kernel", key_kernel},
            {"attention/self/key/bias", key_bias},
            {"attention/self/value/kernel", value_kernel},
            {"attention/self/value/bias", value_bias},
    };

    size_t batch_size = 2;
    size_t num_attention_heads = 2;
    size_t size_per_head = 3;
    size_t seq_length = 4;

    float tensor[48] = {0, 1, 2, 3, 4, 5,
                        6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47};

    float mask[64];
    std::fill_n(mask, 64, 0);
    mask[0] = 1;

    float out[48];

    // compute
    AttentionSelf attention_self(cublas, "attention/self", var, 32, seq_length,
                                 out,
                                 num_attention_heads * size_per_head, num_attention_heads, size_per_head);
    attention_self.compute_cpu(batch_size, tensor, mask);

    EXPECT_FLOAT_EQ(out[0], 3.1048317);
    EXPECT_FLOAT_EQ(out[1], 0.9379833);
    EXPECT_FLOAT_EQ(out[2], -2.2327702);
    EXPECT_FLOAT_EQ(out[3], 0.8868696);
    EXPECT_FLOAT_EQ(out[4], -0.2311137);
    EXPECT_FLOAT_EQ(out[5], -0.43307996);
    EXPECT_FLOAT_EQ(out[42], 5.7841063);
    EXPECT_FLOAT_EQ(out[43], 1.9979048);
    EXPECT_FLOAT_EQ(out[44], -4.535029);
    EXPECT_FLOAT_EQ(out[45], 1.191874);
    EXPECT_FLOAT_EQ(out[46], -0.8387809);
    EXPECT_FLOAT_EQ(out[47], -1.3139831);
}
