#ifndef CUBERT_BERTPOOLER_H
#define CUBERT_BERTPOOLER_H

#include <cstddef>

namespace cuBERT {

    template <typename T>
    void tanh_(T *inout,
               const int N,
               void *stream);

    template <typename T>
    void reduce_mean_1(const T *in, T *out,
                       const int batch_size, const int seq_length, const int hidden_size,
                       void *stream);

    template <typename T>
    class Pooler {
    public:
        /**
         * @param batch_size
         * @param in (batch_size, seq_length, hidden_size)
         * @param out (batch_size, hidden_size)
         */
        virtual void compute(size_t batch_size, T *in, T *out) = 0;
    };

    template <typename T>
    class BertPooler : public Pooler<T> {
    public:
        explicit BertPooler(void* handle,
                            size_t seq_length, size_t hidden_size,
                            T *kernel,
                            T *bias,
                            size_t max_batch_size);

        ~BertPooler();

        void compute(size_t batch_size, T *input_gpu, T *output_gpu) override;

    private:
        void* handle;

        size_t hidden_size;
        size_t seq_length;
        int algo;

        // cpu/gpu buffer
        T *kernel;
        T *bias;
    };

    /**
     * output_layer = tf.reduce_mean(model.get_sequence_output(), axis=1)
     */
    template <typename T>
    class MeanPooler : public Pooler<T> {
    public:
        explicit MeanPooler(void *handle,
                                size_t seq_length, size_t hidden_size);

        ~MeanPooler() = default;

        void compute(size_t batch_size, T *in, T *out) override;

    private:
        void* handle;

        size_t hidden_size;
        size_t seq_length;
    };
}

#endif //CUBERT_BERTPOOLER_H
