#ifndef CUBERT_ADDITIONALOUTPUTLAYER_H
#define CUBERT_ADDITIONALOUTPUTLAYER_H


#include <cublas_v2.h>

namespace cuBERT {
    class AdditionalOutputLayer {
    public:
        explicit AdditionalOutputLayer(cublasHandle_t handle, size_t hidden_size, float *output_weights);

        virtual ~AdditionalOutputLayer();

        void compute(size_t batch_size, float *in_gpu, float *out_gpu);

    private:
        cublasHandle_t handle;

        size_t hidden_size;

        float *output_weights_gpu;
    };
}

#endif //CUBERT_ADDITIONALOUTPUTLAYER_H
