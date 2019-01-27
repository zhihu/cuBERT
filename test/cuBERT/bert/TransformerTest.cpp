//
// Created by 田露 on 2019/1/22.
//

#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <algorithm>

#include "cuBERT/bert/Transformer.h"
using namespace cuBERT;

class TransformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cublasCreate_v2(&cublas);
        cudnnCreate(&cudnn);
    }

    void TearDown() override {
        cublasDestroy_v2(cublas);
        cudnnDestroy(cudnn);
    }

    cublasHandle_t cublas;
    cudnnHandle_t cudnn;

    float query_kernel[36] = {-0.07848196, -0.18097023, 0.06933199, -0.07760319, 0.11389876, 0.05236414,
                              -0.02015782, 0.00233333, -0.00281469, -0.01525305, 0.17362033, -0.01600084,
                              0.00521428, 0.06063714, -0.10533229, 0.0228875, -0.00108843, -0.05974746,
                              -0.05530503, 0.06056419, 0.099603, 0.04929306, 0.08636444, 0.08424559,
                              0.02739674, -0.08676406, -0.0819858, 0.03834791, -0.03903558, 0.01903536,
                              0.01325864, 0.07587593, 0.20709228, -0.0421985, -0.10500058, -0.08004139};
    float query_bias[6] = {-0.01566293, -0.01429354, -0.02946532, 0.02332242, -0.03551506, 0.00519018};

    float key_kernel[36] = {-0.19046976, -0.052291, 0.00774184, -0.04793982, -0.03272828, -0.07022775,
                            0.05397043, 0.22157724, -0.28796428, -0.13628182, 0.10769557, -0.04396444,
                            0.11023977, 0.11277004, -0.17019109, -0.00998783, -0.13503011, 0.03862515,
                            -0.00570178, -0.03683843, -0.09878516, -0.08536254, -0.20706373, 0.07736684,
                            0.09753255, 0.08549864, 0.07573727, -0.08459285, 0.11262332, -0.06660723,
                            -0.05978908, 0.04687774, 0.20048976, -0.15552515, -0.09287686, -0.05736409};
    float key_bias[6] = {0.01119683, -0.00749641, 0.00929781, -0.00789247, 0.00374282, -0.0203852};

    float value_kernel[36] = {0.18298741, 0.13052676, 0.13003705, -0.07762788, -0.11298412, -0.09672086,
                              -0.27567647, -0.11159269, -0.20191047, -0.04961415, 0.03338585, -0.00217377,
                              0.0080993, -0.0092568, -0.07923323, -0.09595821, -0.0724212, 0.00234286,
                              0.08350474, 0.10685625, -0.03265393, 0.12026393, 0.11865459, 0.03879681,
                              0.09247954, -0.08354547, -0.04044447, 0.05576184, 0.063286, -0.06426957,
                              0.11189654, 0.04743394, 0.04952021, 0.06824017, -0.0718908, 0.06118326};
    float value_bias[6] = {-0.01532887, -0.02567805, 0.02993296, 0.00255634, 0.03075514, -0.02086536};

    float attention_output_kernel[36] = {-0.02547911, 0.04877987, 0.05000711, 0.04084699, -0.08732582, 0.09071281,
                                         -0.04081769, 0.21188675, 0.05063592, 0.04011015, -0.09087955, -0.02277032,
                                         0.11330121, -0.00220912, -0.21545858, -0.0109133, 0.12117786, -0.07627827,
                                         0.03476971, 0.113976, 0.0352498, -0.00169246, 0.17134688, 0.05991947,
                                         -0.04367283, -0.08021438, 0.07809242, 0.04896554, -0.09109284, -0.17430527,
                                         0.07785448, -0.08642721, 0.0911883, -0.00432356, -0.10407569, 0.03155923};
    float attention_output_bias[6] = {0.00502381, 0.00164522, 0.00503161, 0.05414474, 0.00594567, -0.00136505};

    float attention_norm_beta[6] = {0.01438165, 0.00893282, -0.00166658, -0.01515444, 0.01131669, -0.00312567};
    float attention_norm_gamma[6] = {0.98945833, 1.00672382, 1.00227484, 0.98692834, 1.00251162, 0.99780415};

    float intermediate_kernel[30] = {0.04110776, 0.00867842, -0.11692518, 0.00942204, 0.00212334,
                                     0.03458865, -0.00608362, -0.12785568, 0.01738149, -0.0735809,
                                     -0.03358123, -0.02204291, 0.19460295, 0.10060768, -0.11971488,
                                     0.02828389, -0.07767208, 0.03127521, 0.01363018, -0.14119004,
                                     0.01852505, -0.12854275, 0.0481119, -0.15679542, -0.08593457,
                                     0.00225799, -0.03674033, -0.10633834, 0.03639213, 0.07383945};
    float intermediate_bias[5] = {-0.0094941, 0.00329734, 0.00365913, 0.02430543, 0.04413794};

    float output_kernel[30] = {0.14096574, 0.0019019, 0.03194073, -0.01783772, 0.04542776, -0.17121975,
                               -0.03054714, -0.03382285, -0.14785342, -0.04588855, -0.09048948, -0.04335051,
                               0.12839685, -0.17706056, -0.01360187, 0.02532171, 0.08845975, 0.00350385,
                               0.07184936, 0.11032352, 0.0339272, -0.04756412, -0.20521204, 0.12666636,
                               0.06397831, -0.15246845, -0.00572673, -0.09259837, -0.00063671, -0.13432225};
    float output_bias[6] = {-0.01755394, 0.02878171, 0.04216052, 0.01562296, 0.01129209, 0.04988396};

    float output_norm_beta[6] = {0.00422856, 0.04091637, 0.03255221, -0.03470522, 0.01916321, -0.00184435};
    float output_norm_gamma[6] = {0.9903053, 0.95159506, 0.98762059, 0.99406842, 1.00686035, 0.97648946};
};

TEST_F(TransformerTest, compute) {
    std::unordered_map<std::string, float *> var = {
            {"encoder/layer_0/attention/self/query/kernel",      query_kernel},
            {"encoder/layer_0/attention/self/query/bias",        query_bias},
            {"encoder/layer_0/attention/self/key/kernel",        key_kernel},
            {"encoder/layer_0/attention/self/key/bias",          key_bias},
            {"encoder/layer_0/attention/self/value/kernel",      value_kernel},
            {"encoder/layer_0/attention/self/value/bias",        value_bias},
            {"encoder/layer_0/attention/output/dense/kernel",    attention_output_kernel},
            {"encoder/layer_0/attention/output/dense/bias",      attention_output_bias},
            {"encoder/layer_0/attention/output/LayerNorm/beta",  attention_norm_beta},
            {"encoder/layer_0/attention/output/LayerNorm/gamma", attention_norm_gamma},
            {"encoder/layer_0/intermediate/dense/kernel",        intermediate_kernel},
            {"encoder/layer_0/intermediate/dense/bias",          intermediate_bias},
            {"encoder/layer_0/output/dense/kernel",              output_kernel},
            {"encoder/layer_0/output/dense/bias",                output_bias},
            {"encoder/layer_0/output/LayerNorm/beta",            output_norm_beta},
            {"encoder/layer_0/output/LayerNorm/gamma",           output_norm_gamma},
    };

    size_t batch_size = 2;
    size_t num_attention_heads = 2;
    size_t size_per_head = 3;
    size_t seq_length = 4;
    size_t intermediate_size = 5;

    size_t hidden_size = num_attention_heads * size_per_head;

    Transformer transformer(cublas, cudnn, "encoder", var, 32,
                            seq_length, hidden_size, 1, num_attention_heads, intermediate_size);

    float tensor[48] = {0, 1, 2, 3, 4, 5,
                        6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47};

    char mask[8];
    std::fill_n(mask, 8, 1);
    mask[0] = 0;

    float out[48];

    // gpu
    float *tensor_gpu;
    char *mask_gpu;
    cudaMalloc(&tensor_gpu, sizeof(float) * 48);
    cudaMalloc(&mask_gpu, sizeof(char) * 8);
    cudaMemcpy(tensor_gpu, tensor, sizeof(float) * 48, cudaMemcpyHostToDevice);
    cudaMemcpy(mask_gpu, mask, sizeof(char) * 8, cudaMemcpyHostToDevice);

    // compute
    float *out_gpu = transformer.compute(batch_size, tensor_gpu, mask_gpu);

    cudaMemcpy(out, out_gpu, sizeof(float) * 48, cudaMemcpyDeviceToHost);
    cudaFree(tensor_gpu);
    cudaFree(mask_gpu);

    EXPECT_FLOAT_EQ(out[0], -1.655961);
    EXPECT_FLOAT_EQ(out[1], -0.5762695);
    EXPECT_FLOAT_EQ(out[2], 0.019856449);
    EXPECT_FLOAT_EQ(out[3], 0.22128667);
    EXPECT_FLOAT_EQ(out[4], 0.5440447);
    EXPECT_FLOAT_EQ(out[5], 1.5205542);
    EXPECT_FLOAT_EQ(out[42], -1.7647616);
    EXPECT_FLOAT_EQ(out[43], -0.35221404);
    EXPECT_FLOAT_EQ(out[44], 0.24000001);
    EXPECT_FLOAT_EQ(out[45], 0.19876797);
    EXPECT_FLOAT_EQ(out[46], 0.19049276);
    EXPECT_FLOAT_EQ(out[47], 1.5452648);
}

TEST_F(TransformerTest, compute_complex) {
    std::unordered_map<std::string, float *> var = {
            {"encoder/layer_0/attention/self/query/kernel",      query_kernel},
            {"encoder/layer_0/attention/self/query/bias",        query_bias},
            {"encoder/layer_0/attention/self/key/kernel",        key_kernel},
            {"encoder/layer_0/attention/self/key/bias",          key_bias},
            {"encoder/layer_0/attention/self/value/kernel",      value_kernel},
            {"encoder/layer_0/attention/self/value/bias",        value_bias},
            {"encoder/layer_0/attention/output/dense/kernel",    attention_output_kernel},
            {"encoder/layer_0/attention/output/dense/bias",      attention_output_bias},
            {"encoder/layer_0/attention/output/LayerNorm/beta",  attention_norm_beta},
            {"encoder/layer_0/attention/output/LayerNorm/gamma", attention_norm_gamma},
            {"encoder/layer_0/intermediate/dense/kernel",        intermediate_kernel},
            {"encoder/layer_0/intermediate/dense/bias",          intermediate_bias},
            {"encoder/layer_0/output/dense/kernel",              output_kernel},
            {"encoder/layer_0/output/dense/bias",                output_bias},
            {"encoder/layer_0/output/LayerNorm/beta",            output_norm_beta},
            {"encoder/layer_0/output/LayerNorm/gamma",           output_norm_gamma},
            {"encoder/layer_1/attention/self/query/kernel",      query_kernel},
            {"encoder/layer_1/attention/self/query/bias",        query_bias},
            {"encoder/layer_1/attention/self/key/kernel",        key_kernel},
            {"encoder/layer_1/attention/self/key/bias",          key_bias},
            {"encoder/layer_1/attention/self/value/kernel",      value_kernel},
            {"encoder/layer_1/attention/self/value/bias",        value_bias},
            {"encoder/layer_1/attention/output/dense/kernel",    attention_output_kernel},
            {"encoder/layer_1/attention/output/dense/bias",      attention_output_bias},
            {"encoder/layer_1/attention/output/LayerNorm/beta",  attention_norm_beta},
            {"encoder/layer_1/attention/output/LayerNorm/gamma", attention_norm_gamma},
            {"encoder/layer_1/intermediate/dense/kernel",        intermediate_kernel},
            {"encoder/layer_1/intermediate/dense/bias",          intermediate_bias},
            {"encoder/layer_1/output/dense/kernel",              output_kernel},
            {"encoder/layer_1/output/dense/bias",                output_bias},
            {"encoder/layer_1/output/LayerNorm/beta",            output_norm_beta},
            {"encoder/layer_1/output/LayerNorm/gamma",           output_norm_gamma},
    };

    size_t batch_size = 2;
    size_t num_attention_heads = 2;
    size_t size_per_head = 3;
    size_t seq_length = 4;
    size_t intermediate_size = 5;

    size_t hidden_size = num_attention_heads * size_per_head;

    Transformer transformer(cublas, cudnn, "encoder", var, 32,
                            seq_length, hidden_size, 2, num_attention_heads, intermediate_size);

    float tensor[48] = {0, 1, 2, 3, 4, 5,
                        6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47};

    char mask[8];
    std::fill_n(mask, 8, 1);
    mask[0] = 0;

    float out[48];

    // gpu
    float *tensor_gpu;
    char *mask_gpu;
    cudaMalloc(&tensor_gpu, sizeof(float) * 48);
    cudaMalloc(&mask_gpu, sizeof(char) * 8);
    cudaMemcpy(tensor_gpu, tensor, sizeof(float) * 48, cudaMemcpyHostToDevice);
    cudaMemcpy(mask_gpu, mask, sizeof(char) * 8, cudaMemcpyHostToDevice);

    // compute
    float *out_gpu = transformer.compute(batch_size, tensor_gpu, mask_gpu);

    cudaMemcpy(out, out_gpu, sizeof(float) * 48, cudaMemcpyDeviceToHost);
    cudaFree(tensor_gpu);
    cudaFree(mask_gpu);

    EXPECT_NEAR(out[0], -1.6749241, 1e-5);
    EXPECT_NEAR(out[1], -0.5803416, 1e-5);
    EXPECT_NEAR(out[2], 0.07921408, 1e-5);
    EXPECT_NEAR(out[3], 0.18109897, 1e-5);
    EXPECT_NEAR(out[4], 0.5763312, 1e-5);
    EXPECT_NEAR(out[5], 1.4929073, 1e-5);
    EXPECT_NEAR(out[42], -1.7720084, 1e-5);
    EXPECT_NEAR(out[43], -0.3701498, 1e-5);
    EXPECT_NEAR(out[44], 0.3161025, 1e-5);
    EXPECT_NEAR(out[45], 0.15598607, 1e-5);
    EXPECT_NEAR(out[46], 0.20395504, 1e-5);
    EXPECT_NEAR(out[47], 1.5245408, 1e-5);
}
