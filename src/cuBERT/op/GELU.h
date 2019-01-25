//
// Created by 田露 on 2019/1/21.
//

#ifndef CUBERT_GELU_H
#define CUBERT_GELU_H


#include <cuda_runtime.h>

namespace cuBERT {
    __host__ void gelu_(float *inout,
                        const int N,
                        cudaStream_t stream);

    class GELU {
    public:
        void compute_(size_t N, float *inout_gpu, cudaStream_t stream);

        void compute_cpu_(size_t N, float *inout, cudaStream_t stream);
    };
}

#endif //CUBERT_GELU_H
