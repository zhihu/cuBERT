#ifndef CUBERT_GELU_H
#define CUBERT_GELU_H


#include <cuda_runtime.h>

namespace cuBERT {

    class GELU {
    public:
        void compute_(size_t N, float *inout_gpu, cudaStream_t stream);

        void compute_cpu_(size_t N, float *inout);
    };
}

#endif //CUBERT_GELU_H
