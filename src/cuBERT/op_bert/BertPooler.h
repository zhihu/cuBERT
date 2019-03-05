#ifndef CUBERT_BERTPOOLER_H
#define CUBERT_BERTPOOLER_H

#include <cstddef>

namespace cuBERT {

    template <bool cpu>
    void tanh_(float *inout,
               const int N,
               void *stream);

    template <bool cpu>
    void reduce_mean_1(const float *in, float *out,
                       const int batch_size, const int seq_length, const int hidden_size,
                       void *stream);

    class Pooler {
    public:
        /**
         * @param batch_size
         * @param in (batch_size, seq_length, hidden_size)
         * @param out (batch_size, hidden_size)
         */
        virtual void compute(size_t batch_size, float *in, float *out) = 0;
    };

    class BertPooler : public Pooler {
    public:
        explicit BertPooler(void* handle,
                            size_t seq_length, size_t hidden_size,
                            float *kernel,
                            float *bias,
                            size_t max_batch_size);

        virtual ~BertPooler();

        void compute(size_t batch_size, float *input_gpu, float *output_gpu) override;

    private:
        void* handle;

        size_t hidden_size;
        size_t seq_length;

        // cpu/gpu buffer
        float *kernel;
        float *bias;
    };

    /**
     * tf.reduce_mean(model.all_encoder_layers[-1], axis=1)
     */
    class MeanPooler : public Pooler {
    public:
        explicit MeanPooler(void *handle,
                                size_t seq_length, size_t hidden_size);

        virtual ~MeanPooler() = default;

        void compute(size_t batch_size, float *in, float *out) override;

    private:
        void* handle;

        size_t hidden_size;
        size_t seq_length;
    };
}

#endif //CUBERT_BERTPOOLER_H
