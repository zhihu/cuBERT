#ifndef CUBERT_ADDITIONALOUTPUTLAYER_H
#define CUBERT_ADDITIONALOUTPUTLAYER_H


#include <cstddef>


#include "cuBERT/op/Softmax.h"

namespace cuBERT {

    /**
     * output_layer = model.get_pooled_output()
     *
     * output_weights = tf.get_variable(
     *   "output_weights", [num_labels, hidden_size],
     *   initializer=tf.truncated_normal_initializer(stddev=0.02))
     * 
     * output_bias = tf.get_variable(
     *   "output_bias", [num_labels], initializer=tf.zeros_initializer())
     *
     * logits = tf.matmul(output_layer, output_weights, transpose_b=True)
     * logits = tf.nn.bias_add(logits, output_bias)
     */
    template <typename T>
    class ClassifierOutputLayer {
    public:
        explicit ClassifierOutputLayer(void* handle, 
                                       size_t hidden_size, 
                                       size_t num_labels, 
                                       T *output_weights, 
                                       T *output_bias, 
                                       size_t max_batch_size);

        virtual ~ClassifierOutputLayer();

        void _pre_compute(size_t batch_size, T *output);

        void _in_compute(size_t batch_size, T *input, T *output);

        void compute(size_t batch_size, T *in_gpu, T *out_gpu);

    private:
        void* handle;

        size_t hidden_size;
        size_t num_labels;

        // cpu/gpu buffer
        T *output_weights;
        T *output_bias;
    };
}

#endif //CUBERT_ADDITIONALOUTPUTLAYER_H
