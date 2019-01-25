//
// Created by 田露 on 2019/1/18.
//

#ifndef CUBERT_SOFTMAX_H
#define CUBERT_SOFTMAX_H


#include <cudnn.h>

namespace cuBERT {
/**
 * Performance is expected to be highest with NCHW fully-packed tensors.
 */
    class Softmax {
    public:
        explicit Softmax(cudnnHandle_t handle, size_t channel);

        virtual ~Softmax();

        void compute_(size_t batch_size, float *inout_gpu);

    private:
        cudnnHandle_t handle;

        size_t channel;

        cudnnTensorDescriptor_t desc;
    };
}

#endif //CUBERT_SOFTMAX_H
