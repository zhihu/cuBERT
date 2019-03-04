#ifndef CUBERT_LAYERNORM_H
#define CUBERT_LAYERNORM_H

namespace cuBERT {

    template <bool cpu>
    void layer_norm_(float *inout,
                     const int batch_size,
                     const int channels,
                     const float *beta,
                     const float *gamma,
                     void *stream);

    template <bool cpu>
    void layer_norm_(const float *in,
                     float *inout,
                     const int batch_size,
                     const int channels,
                     float *mean_gpu,
                     float *var_gpu,
                     const float *beta,
                     const float *gamma,
                     void *stream);

    class LayerNorm {
    public:
        explicit LayerNorm(size_t max_batch_size, size_t channels, float *beta, float *gamma);

        virtual ~LayerNorm();

        void compute_(size_t batch_size, float *inout_gpu, void* stream);

        void compute_(size_t batch_size, float *in_gpu, float *inout_gpu, void* stream);

    private:
        size_t channels;

        // cpu/gpu buffer
        float *beta;
        float *gamma;

        float *mean_gpu;
        float *var_gpu;
    };
}

#endif //CUBERT_LAYERNORM_H
