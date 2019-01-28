#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuBERT/common.h"
#include "GELU.h"

namespace cuBERT {

    struct gelu_functor {
        __device__ float operator()(const float& x) const {
            return x * 0.5f * (1.0f + erff(x * sqrtf(0.5)));
        }
    };

    void GELU::compute_(size_t N, float *inout_gpu, cudaStream_t stream) {
        thrust::device_ptr<float> dev_ptr(inout_gpu);
        thrust::transform(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + N, dev_ptr, gelu_functor());
    }

    void GELU::compute_cpu_(size_t N, float *inout, cudaStream_t stream) {
        float *inout_gpu;
        cudaMalloc(&inout_gpu, sizeof(float) * N);

        CUDA_CHECK(cudaMemcpyAsync(inout_gpu, inout, sizeof(float) * N, cudaMemcpyHostToDevice, stream));
        compute_(N, inout_gpu, stream);

        // sync
        CUDA_CHECK(cudaMemcpy(inout, inout_gpu, sizeof(float) * N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(inout_gpu));
    }
}
