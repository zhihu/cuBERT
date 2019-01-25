#ifndef CUBERT_BERTPOOLER_H
#define CUBERT_BERTPOOLER_H

#include <cublas_v2.h>

namespace cuBERT {
    __host__ void tanh_(float *inout,
                        const int N,
                        cudaStream_t stream);


    class BertPooler {
    public:
        explicit BertPooler(cublasHandle_t handle,
                            size_t seq_length, size_t hidden_size,
                            float *kernel,
                            float *bias,
                            size_t max_batch_size);

        virtual ~BertPooler();

        void compute(size_t batch_size, float *input_gpu, float *output_gpu);

    private:
        cublasHandle_t handle;

        size_t hidden_size;
        size_t seq_length;

        float *kernel_gpu;
        float *bias_gpu;
    };
}

#endif //CUBERT_BERTPOOLER_H
