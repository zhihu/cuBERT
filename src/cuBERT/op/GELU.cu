#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

#include "math.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "GELU.h"

namespace cuBERT {

    template <typename T>
    struct gelu_functor {
        __device__ T operator()(const T& x) const {
            float _x = (float) x;
            float _y = _x * 0.5f * (1.0f + erff(_x * sqrtf(0.5f)));
            return T(_y);
        }
    };

    template <typename T>
    __host__ void gelu(size_t N, T *inout_gpu, void *stream) {
        thrust::device_ptr<T> dev_ptr(inout_gpu);
        thrust::transform(thrust::cuda::par.on((cudaStream_t) stream), dev_ptr, dev_ptr + N, dev_ptr, gelu_functor<T>());
    }

    template
    __host__ void gelu<float>(size_t N, float *inout_gpu, void *stream);

    template
    __host__ void gelu<half>(size_t N, half *inout_gpu, void *stream);
}
