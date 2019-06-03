#include "cuBERT/common.h"
#include "AdditionalOutputLayer.h"

namespace cuBERT {

    template <typename T>
    ClassifierOutputLayer<T>::ClassifierOutputLayer(void* handle, 
                                                    size_t hidden_size, 
                                                    size_t num_labels, 
                                                    T *output_weights, 
                                                    T *output_bias, 
                                                    size_t max_batch_size) {
        this->handle = handle;
        this->hidden_size = hidden_size;
        this->num_labels = num_labels;

        this->output_weights = static_cast<T *>(cuBERT::malloc(sizeof(T) * hidden_size * this->num_labels));
        cuBERT::memcpy(this->output_weights, output_weights, sizeof(T) * hidden_size * this->num_labels, 1);

        if (output_bias != nullptr) {
            this->output_bias = static_cast<T *>(cuBERT::malloc(sizeof(T) * num_labels * max_batch_size));
            for (int i = 0; i < max_batch_size; ++i) {
                cuBERT::memcpy(this->output_bias + num_labels * i, output_bias, num_labels * sizeof(T), 1);
            }
        } else {
            this->output_bias = nullptr;
        }

        this->softmax = new Softmax<T>(max_batch_size, num_labels);
    }

    template <typename T>
    ClassifierOutputLayer<T>::~ClassifierOutputLayer() {
        delete softmax;

        if (output_bias != nullptr) {
            cuBERT::free(output_bias);
        }
        cuBERT::free(output_weights);
    }

    template<typename T>
    void ClassifierOutputLayer<T>::_pre_compute(size_t batch_size, T *output) {
        if (output_bias != nullptr) {
            void* streamId = blas_get_stream(handle);
            cuBERT::memcpyAsync(output, output_bias, num_labels * batch_size * sizeof(T), 3, streamId);
        }
    }

    template<typename T>
    void ClassifierOutputLayer<T>::_in_compute(size_t batch_size, T *input, T *output_logits, T *output_probs) {
        float beta = output_bias == nullptr ? 0.f : 1.f;
        cuBERT::blas_gemm(handle, true, false,
                          num_labels, batch_size, hidden_size,
                          1.f,
                          output_weights, hidden_size,
                          input, hidden_size,
                          beta,
                          output_logits, num_labels);
        if (output_probs != nullptr) {
            void* streamId = blas_get_stream(handle);
            softmax->compute_(batch_size, output_logits, output_probs, streamId);
        }
    }

    template <typename T>
    void ClassifierOutputLayer<T>::compute(size_t batch_size, T *input, T *output_logits, T *output_probs) {
        _pre_compute(batch_size, output_logits);
        _in_compute(batch_size, input, output_logits, output_probs);
    }

    template class ClassifierOutputLayer<float>;
#ifdef HAVE_CUDA
    template class ClassifierOutputLayer<half>;
#endif
}
