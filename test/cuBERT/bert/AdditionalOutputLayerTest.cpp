#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuBERT/bert/AdditionalOutputLayer.h"
using namespace cuBERT;

class AdditionalOutputLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cublasCreate_v2(&handle);
    }

    void TearDown() override {
        cublasDestroy_v2(handle);
    }

    cublasHandle_t handle;
};


TEST_F(AdditionalOutputLayerTest, compute) {
    size_t hidden_size = 3;
    float output_weights[3] = {-1, 0, 1};

    AdditionalOutputLayer aol(handle, hidden_size, output_weights);

    float in[6] = {
            2, 8, 3,
            -4, 1, 2,
    };
    float out[2];

    float* in_gpu;
    float* out_gpu;
    cudaMalloc(&in_gpu, sizeof(float) * 6);
    cudaMalloc(&out_gpu, sizeof(float) * 2);

    cudaMemcpy(in_gpu, in, sizeof(float) * 6, cudaMemcpyHostToDevice);

    aol.compute(2, in_gpu, out_gpu);

    cudaMemcpy(out, out_gpu, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    cudaFree(in_gpu);
    cudaFree(out_gpu);

    EXPECT_FLOAT_EQ(out[0], 1);
    EXPECT_FLOAT_EQ(out[1], 6);
}
