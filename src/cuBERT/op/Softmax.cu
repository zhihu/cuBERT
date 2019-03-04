#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

#include "math.h"
#include "cub/cub.cuh"
#include <cuda_runtime.h>

namespace cuBERT {

    struct exp_functor {
        __device__ float operator()(const float& x) const {
            return __expf(x);
        }
    };

    __global__ void kernel_sum_cub(const float *__restrict__ in,
                                   const int batch_size,
                                   const int channel,
                                   float *sum_out) {
        __shared__ typename cub::BlockReduce<float, 128>::TempStorage s_storage;
        for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
            float s_val = 0;
            for (int j = threadIdx.x; j < channel; j += blockDim.x) {
                s_val += __ldg(in + i * channel + j);
            }
            s_val = cub::BlockReduce<float, 128>(s_storage).Sum(s_val);
            if (threadIdx.x == 0) {
                sum_out[i] = s_val;
            }
            __syncthreads();
        }
    }

    __global__ void kernel_scale_(float *inout, const int batch_size, const int channel, float *sum_in) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * channel) {
            return;
        }

        int batch_idx = idx / channel;
        inout[idx] = __ldg(inout + idx) / __ldg(sum_in + batch_idx);
    }

    template <bool cpu>
    __host__ void softmax_(float *inout, const int batch_size, const int channel, float *sum_gpu, void* stream) {
        thrust::device_ptr<float> dev_ptr(inout);
        thrust::transform(thrust::cuda::par.on((cudaStream_t) stream), dev_ptr, dev_ptr + batch_size * channel, dev_ptr, exp_functor());

        const int all_blocks = (batch_size * channel + 127) / 128;
        kernel_sum_cub <<<batch_size, 128, 0, (cudaStream_t) stream>>> (inout, batch_size, channel, sum_gpu);
        kernel_scale_ <<<all_blocks, 128, 0, (cudaStream_t) stream>>> (inout, batch_size, channel, sum_gpu);
    }

    template
    __host__ void softmax_<false>(float *inout, const int batch_size, const int channel, float *sum_gpu, void *stream);
}
