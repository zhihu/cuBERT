//
// Created by 田露 on 2019/1/15.
//

#ifndef CUBERT_LAYERNORM_H
#define CUBERT_LAYERNORM_H

#include <cuda_runtime.h>


namespace cuBERT {
    __host__ void layer_norm_(float *inout,
                              const int batch_size,
                              const int channel,
                              float *beta,
                              float *gamma,
                              cudaStream_t stream);

    __host__ void layer_norm_(float *in,
                              float *inout,
                              const int batch_size,
                              const int channel,
                              float *beta,
                              float *gamma,
                              cudaStream_t stream);

    class LayerNorm {
    public:
        explicit LayerNorm(size_t channels, float *beta, float *gamma);

        virtual ~LayerNorm();

        void compute_(size_t batch_size, float *inout_gpu, cudaStream_t stream);

        void compute_(size_t batch_size, float *in_gpu, float *inout_gpu, cudaStream_t stream);

        void compute_cpu_(size_t batch_size, float *inout, cudaStream_t stream);

        void compute_cpu_(size_t batch_size, float *in, float *inout, cudaStream_t stream);

    private:
        size_t channels;
        float *beta_gpu;
        float *gamma_gpu;
    };
}

#endif //CUBERT_LAYERNORM_H
