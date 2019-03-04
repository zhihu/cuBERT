#include "math.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

namespace cuBERT {
    struct tanh_functor {
        __device__ float operator()(const float& x) const {
            return tanhf(x);
        }
    };

    template <bool cpu>
    __host__ void tanh_(float *inout,
                        const int N,
                        void *stream) {
        thrust::device_ptr<float> dev_ptr(inout);
        thrust::transform(thrust::cuda::par.on((cudaStream_t) stream), dev_ptr, dev_ptr + N, dev_ptr, tanh_functor());
    }

    template
    __host__ void tanh_<false>(float *inout,
                               const int N,
                               void *stream);
}
