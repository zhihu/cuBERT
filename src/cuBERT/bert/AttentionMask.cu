#include <cuda_runtime.h>

namespace cuBERT {
    __global__ void kernel_not(const char *__restrict__ in,
                               float *out,
                               const int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) {
            return;
        }

        out[idx] = (float) !__ldg(in + idx);
    }

    __host__ void _not(const char *in,
                       float *out,
                       const int N,
                       cudaStream_t stream) {
        const int blocks = (N + 127) / 128;
        kernel_not << < blocks, 128, 0, stream >> > (in, out, N);
    }
}
