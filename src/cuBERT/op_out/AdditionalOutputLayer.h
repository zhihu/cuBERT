#ifndef CUBERT_ADDITIONALOUTPUTLAYER_H
#define CUBERT_ADDITIONALOUTPUTLAYER_H


#include <cstddef>

namespace cuBERT {

    /**
     * output_layer = model.get_pooled_output()
     *
     * output_weights = tf.get_variable(
     *   "output_weights", [1, hidden_size],
     *   initializer=tf.truncated_normal_initializer(stddev=0.02))
     *
     * logits = tf.matmul(output_layer, output_weights, transpose_b=True)
     */
    template <typename T>
    class AdditionalOutputLayer {
    public:
        explicit AdditionalOutputLayer(void* handle, size_t hidden_size, T *output_weights);

        virtual ~AdditionalOutputLayer();

        void compute(size_t batch_size, T *in_gpu, T *out_gpu);

    private:
        void* handle;

        size_t hidden_size;

        // cpu/gpu buffer
        T *output_weights;
    };
}

#endif //CUBERT_ADDITIONALOUTPUTLAYER_H
