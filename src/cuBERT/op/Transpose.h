//
// Created by 田露 on 2019/1/17.
//

#ifndef CUBERT_TRANSPOSE_H
#define CUBERT_TRANSPOSE_H

#include <vector>
#include <cudnn.h>

namespace cuBERT {
/**
 * Dim transpose for >= 4D tensor. from (batch_size, ...) to (batch_size, ...).
 * The first dim should not be modified, and always set it to -1 in the constructor.
 */
    class Transpose {
    public:
        explicit Transpose(cudnnHandle_t handle,
                           const std::vector<int> &dims_in,
                           const std::vector<int> &axes);

        virtual ~Transpose();

        void compute(size_t batch_size, float *in_gpu, float *out_gpu);

        void compute_cpu(size_t batch_size, float *in, float *out);

    private:
        cudnnHandle_t handle;

        std::vector<int> dims_in;
        std::vector<int> dims_out;
        std::vector<int> stride_in;
        std::vector<int> stride_out;

        cudnnTensorDescriptor_t desc_in;
        cudnnTensorDescriptor_t desc_out;
    };
}

#endif //CUBERT_TRANSPOSE_H
