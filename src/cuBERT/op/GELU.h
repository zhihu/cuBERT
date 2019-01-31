#ifndef CUBERT_GELU_H
#define CUBERT_GELU_H

namespace cuBERT {

    void gelu(size_t N, float *inout_gpu, void *stream);

    class GELU {
    public:
        void compute_(size_t N, float *inout_gpu, void* stream);
    };
}

#endif //CUBERT_GELU_H
