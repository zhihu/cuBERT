#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>

#include "AttentionMask.h"

namespace cuBERT {
    template<typename T>
    __host__ void _not(const int8_t *in,
                       T *out,
                       const int N,
                       void *stream) {
        thrust::device_ptr<const int8_t> in_ptr(in);
        thrust::device_ptr<T> out_ptr(out);
        thrust::transform(thrust::cuda::par.on((cudaStream_t) stream), in_ptr, in_ptr + N, out_ptr, thrust::logical_not<const int8_t>());
    }

    template
    __host__ void _not<float>(const int8_t *in,
                              float *out,
                              const int N,
                              void *stream);

    template
    __host__ void _not<half >(const int8_t *in,
                              half *out,
                              const int N,
                              void *stream);
}
