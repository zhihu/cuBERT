#ifndef CUBERT_BERTPOOLER_H
#define CUBERT_BERTPOOLER_H

namespace cuBERT {
    void tanh_(float *inout,
               const int N,
               void *stream);


    class BertPooler {
    public:
        explicit BertPooler(void* handle,
                            size_t seq_length, size_t hidden_size,
                            float *kernel,
                            float *bias,
                            size_t max_batch_size);

        virtual ~BertPooler();

        void compute(size_t batch_size, float *input_gpu, float *output_gpu);

    private:
        void* handle;

        size_t hidden_size;
        size_t seq_length;

        // cpu/gpu buffer
        float *kernel;
        float *bias;
    };
}

#endif //CUBERT_BERTPOOLER_H
