#ifndef CUBERT_LAYERNORM_H
#define CUBERT_LAYERNORM_H

namespace cuBERT {
    template <typename T>
    void layer_norm_(const T *in,
                     T *inout,
                     const int batch_size,
                     const int channels,
                     T *mean_gpu,
                     T *var_gpu,
                     const T *beta,
                     const T *gamma,
                     void *stream);

    template <typename T>
    class LayerNorm {
    public:
        explicit LayerNorm(size_t max_batch_size, size_t channels, T *beta, T *gamma);

        virtual ~LayerNorm();

        void compute_(size_t batch_size, T *in_gpu, T *inout_gpu, void* stream);

    private:
        size_t channels;

        // cpu/gpu buffer
        T *beta;
        T *gamma;

        T *mean_gpu;
        T *var_gpu;
    };
}

#endif //CUBERT_LAYERNORM_H
