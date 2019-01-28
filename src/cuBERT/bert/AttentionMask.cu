#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>

namespace cuBERT {
    __host__ void _not(const char *in,
                       float *out,
                       const int N,
                       cudaStream_t stream) {
        thrust::device_ptr<const char> in_ptr(in);
        thrust::device_ptr<float> out_ptr(out);
        thrust::transform(thrust::cuda::par.on(stream), in_ptr, in_ptr + N, out_ptr, thrust::logical_not<const char>());
    }
}
