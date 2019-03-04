#ifndef CUBERT_SOFTMAX_H
#define CUBERT_SOFTMAX_H

#include <cstddef>

namespace cuBERT {

    template <bool cpu>
    void softmax_(float *inout,
                  const int batch_size,
                  const int channel,
                  float *sum_gpu,
                  void *stream);

    class Softmax {
    public:
        explicit Softmax(size_t max_batch_size, size_t channel);

        virtual ~Softmax();

        void compute_(size_t batch_size, float *inout_gpu, void* stream);

    private:
        size_t channel;

        float* sum_gpu;
    };
}

#endif //CUBERT_SOFTMAX_H
