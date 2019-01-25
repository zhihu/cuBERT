#include "math.h"
#include <cuda_runtime.h>

namespace cuBERT {
    __global__ void kernel_gelu_(float *inout,
                                 const int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) {
            return;
        }

        float input_tensor = __ldg(inout + idx);
        inout[idx] = input_tensor * 0.5f * (1.0f + erff(input_tensor / sqrtf(2.0)));
    }

    __host__ void gelu_(float *inout,
                        const int N,
                        cudaStream_t stream) {
        const int blocks = (N + 127) / 128;
        kernel_gelu_ << < blocks, 128, 0, stream >> > (inout, N);
    }
}
