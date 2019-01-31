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

    __host__ void gelu(size_t N, float *inout_gpu, void *stream) {
        thrust::device_ptr<float> dev_ptr(inout_gpu);
        thrust::transform(thrust::cuda::par.on((cudaStream_t) stream), dev_ptr, dev_ptr + N, dev_ptr, gelu_functor());
    }
}
