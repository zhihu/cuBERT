#include "math.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

#include "BertPooler.h"

namespace cuBERT {
    struct tanh_functor {
        __device__ float operator()(const float& x) const {
            return tanhf(x);
        }
    };

    template <>
    __host__ void tanh_<false>(float *inout,
                               const int N,
                               void *stream) {
        thrust::device_ptr<float> dev_ptr(inout);
        thrust::transform(thrust::cuda::par.on((cudaStream_t) stream), dev_ptr, dev_ptr + N, dev_ptr, tanh_functor());
    }

    __global__ void kernel_reduce_mean_1(const float *__restrict__ in,
                                         float *out,
                                         const int batch_size, const int seq_length, const int channel) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * channel) {
            return;
        }

        int batch_idx = idx / channel;
        int channel_idx = idx % channel;
        const int tmp = channel_idx + batch_idx * seq_length * channel;

        float sum = 0;
#pragma unroll
        for (int seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
            sum += __ldg(in + tmp + seq_idx * channel);
        }

        out[idx] = sum / seq_length;
    }


    template<>
    __host__ void reduce_mean_1<false>(const float *in, float *out,
                                       const int batch_size, const int seq_length, const int hidden_size,
                                       void *stream) {
        const int blocks = (batch_size * hidden_size + 127) / 128;
        kernel_reduce_mean_1 <<<blocks, 128, 0, (cudaStream_t) stream>>> (in, out, batch_size, seq_length, hidden_size);
    }
}
