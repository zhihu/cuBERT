#include "math.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

#include "BertPooler.h"

namespace cuBERT {

    template <typename T>
    struct tanh_functor {
        __device__ T operator()(const T& x) const {
            return T(tanhf((float) x));
        }
    };

    template <typename T>
    __host__ void tanh_(T *inout,
                        const int N,
                        void *stream) {
        thrust::device_ptr<T> dev_ptr(inout);
        thrust::transform(thrust::cuda::par.on((cudaStream_t) stream), dev_ptr, dev_ptr + N, dev_ptr, tanh_functor<T>());
    }

    template
    __host__ void tanh_<float>(float *inout, const int N, void *stream);

    template
    __host__ void tanh_<half>(half *inout, const int N, void *stream);

    template <typename T>
    __global__ void kernel_reduce_mean_1(const T *__restrict__ in,
                                         T *out,
                                         const int batch_size, const int seq_length, const int channel) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * channel) {
            return;
        }

        int batch_idx = idx / channel;
        int channel_idx = idx % channel;
        const int tmp = channel_idx + batch_idx * seq_length * channel;

        float sum = 0.f;
#pragma unroll
        for (int seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
#if __CUDA_ARCH__ >= 350
            sum += (float) __ldg(in + tmp + seq_idx * channel);
#else
            sum += (float) in[tmp + seq_idx * channel];
#endif
        }

        out[idx] = sum / seq_length;
    }

    template <typename T>
    __host__ void reduce_mean_1(const T *in, T *out,
                                const int batch_size, const int seq_length, const int hidden_size,
                                void *stream) {
        const int blocks = (batch_size * hidden_size + 127) / 128;
        kernel_reduce_mean_1<T> <<<blocks, 128, 0, (cudaStream_t) stream>>> (in, out, batch_size, seq_length, hidden_size);
    }

    template
    __host__ void reduce_mean_1<float>(const float *in, float *out,
                                       const int batch_size, const int seq_length, const int hidden_size,
                                       void *stream);

    template
    __host__ void reduce_mean_1<half>(const half *in, half *out,
                                      const int batch_size, const int seq_length, const int hidden_size,
                                      void *stream);
}
