#ifndef CUBERT_GELU_H
#define CUBERT_GELU_H

namespace cuBERT {

    template <typename T>
    void gelu(size_t N, T *inout, void *stream);

    template <typename T>
    class GELU {
    public:
        void compute_(size_t N, T *inout_gpu, void* stream);
    };
}

#endif //CUBERT_GELU_H
