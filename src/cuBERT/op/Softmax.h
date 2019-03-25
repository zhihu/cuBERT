#ifndef CUBERT_SOFTMAX_H
#define CUBERT_SOFTMAX_H

#include <cstddef>

namespace cuBERT {

    template <typename T>
    void softmax_(T *inout,
                  const int batch_size,
                  const int channel,
                  T *sum_gpu,
                  void *stream);

    template <typename T>
    class Softmax {
    public:
        explicit Softmax(size_t max_batch_size, size_t channel);

        virtual ~Softmax();

        void compute_(size_t batch_size, T *inout_gpu, void* stream);

    private:
        size_t channel;

        T* sum_gpu;
    };
}

#endif //CUBERT_SOFTMAX_H
