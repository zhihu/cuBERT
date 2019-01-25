#include "math.h"
#include <cuda_runtime.h>

namespace cuBERT {
    __global__ void kernel_tanh_(float *inout,
                                 const int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) {
            return;
        }

        inout[idx] = tanhf(__ldg(inout + idx));
    }

    __host__ void tanh_(float *inout,
                        const int N,
                        cudaStream_t stream) {
        const int blocks = (N + 127) / 128;
        kernel_tanh_ << < blocks, 128, 0, stream >> > (inout, N);
    }
}
