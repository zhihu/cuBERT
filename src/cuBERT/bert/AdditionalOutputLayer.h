#ifndef CUBERT_ADDITIONALOUTPUTLAYER_H
#define CUBERT_ADDITIONALOUTPUTLAYER_H


#include <cstddef>

namespace cuBERT {
    class AdditionalOutputLayer {
    public:
        explicit AdditionalOutputLayer(void* handle, size_t hidden_size, float *output_weights);

        virtual ~AdditionalOutputLayer();

        void compute(size_t batch_size, float *in_gpu, float *out_gpu);

    private:
        void* handle;

        size_t hidden_size;

        // cpu/gpu buffer
        float *output_weights;
    };
}

#endif //CUBERT_ADDITIONALOUTPUTLAYER_H
