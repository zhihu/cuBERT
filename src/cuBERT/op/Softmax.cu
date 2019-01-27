#include "math.h"
#include "cub/cub.cuh"
#include <cuda_runtime.h>

namespace cuBERT {

__global__ void kernel_exp_(float *inout, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    inout[idx] = __expf(__ldg(inout + idx));
}

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


__host__ void softmax_(float *inout, const int batch_size, const int channel, float *sum_gpu, cudaStream_t stream) {
    const int all_blocks = (batch_size * channel + 127) / 128;
    kernel_exp_ <<<all_blocks, 128, 0, stream>>> (inout, batch_size * channel);

    kernel_sum_cub <<<batch_size, 128, 0, stream>>> (inout, batch_size, channel, sum_gpu);

    kernel_scale_ <<<all_blocks, 128, 0, stream>>> (inout, batch_size, channel, sum_gpu);
}

}
