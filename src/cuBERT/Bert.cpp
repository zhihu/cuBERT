#include <iostream>

#include "cuBERT/common.h"
#include "Bert.h"

namespace cuBERT {
    template <typename T>
    Bert<T>::Bert(const std::unordered_map<std::string, T *> &var,
               size_t max_batch_size,
               size_t seq_length,
               size_t vocab_size,
               size_t type_vocab_size,
               size_t hidden_size,
               size_t num_hidden_layers,
               size_t num_attention_heads,
               size_t intermediate_size,
               size_t num_labels) {
        this->max_batch_size = max_batch_size;
        this->seq_length = seq_length;
        this->hidden_size = hidden_size;
        this->num_labels = num_labels;

        this->stream = cuBERT::cuda_stream_create();
        this->cublas = cuBERT::blas_create();
        cuBERT::blas_set_stream(cublas, stream);

        this->bert_embeddings = new BertEmbeddings<T>(cublas, var, max_batch_size,
                                                   vocab_size, type_vocab_size, hidden_size, seq_length);

        this->transformer = new Transformer<T>(cublas, "bert/encoder", var,
                                            max_batch_size, seq_length,
                                            hidden_size, num_hidden_layers, num_attention_heads, intermediate_size);

        if (var.count("bert/pooler/dense/kernel")) {
            this->bert_pooler = new BertPooler<T>(cublas, seq_length, hidden_size,
                                               var.at("bert/pooler/dense/kernel"),
                                               var.at("bert/pooler/dense/bias"),
                                               max_batch_size);
        } else {
            this->bert_pooler = new MeanPooler<T>(cublas, seq_length, hidden_size);
        }

        if (var.count("output_weights")) {
            T* output_bias = var.count("output_bias") ? var.at("output_bias") : nullptr;
            this->additional_output_layer = new ClassifierOutputLayer<T>(cublas, hidden_size, num_labels, var.at("output_weights"), output_bias, max_batch_size);
        } else {
            this->additional_output_layer = nullptr;
        }

        this->_embedding_output = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * hidden_size));
        this->_pooled_output = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * hidden_size));
        this->_logits = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * num_labels));
        this->_probs = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * num_labels));

        this->input_ids_buf = static_cast<int *>(cuBERT::malloc(sizeof(int) * max_batch_size * seq_length));
        this->input_mask_buf = static_cast<int8_t *>(cuBERT::malloc(sizeof(int8_t) * max_batch_size * seq_length));
        this->segment_ids_buf = static_cast<int8_t *>(cuBERT::malloc(sizeof(int8_t) * max_batch_size * seq_length));

        // pre-compute buffers
        transformer->_pre_compute(max_batch_size);
        if (additional_output_layer != nullptr) {
            additional_output_layer->_pre_compute(max_batch_size, _logits);
        }
        this->buffer_filled = true;
    }

    template <typename T>
    Bert<T>::~Bert() {
        cuBERT::free(segment_ids_buf);
        cuBERT::free(input_mask_buf);
        cuBERT::free(input_ids_buf);

        cuBERT::free(_probs);
        cuBERT::free(_logits);
        cuBERT::free(_pooled_output);
        cuBERT::free(_embedding_output);

        delete additional_output_layer;
        delete bert_pooler;
        delete transformer;
        delete bert_embeddings;

        cuBERT::blas_destroy(cublas);
        cuBERT::cuda_stream_destroy(stream);
    }

    template <typename T>
    void Bert<T>::compute(size_t batch_size, int *input_ids, int8_t *input_mask, int8_t *segment_ids) {
        if (batch_size > max_batch_size) {
            throw std::invalid_argument("batch_size > max_batch_size");	
        } else if (batch_size == 0) {
            return;
        }

#ifdef HAVE_CUDA
        // copy inputs
        void *streamId = blas_get_stream(cublas);
        cuBERT::memcpyAsync(input_ids_buf, input_ids, sizeof(int) * batch_size * seq_length, 1, streamId);
        cuBERT::memcpyAsync(input_mask_buf, input_mask, sizeof(int8_t) * batch_size * seq_length, 1, streamId);
        cuBERT::memcpyAsync(segment_ids_buf, segment_ids, sizeof(int8_t) * batch_size * seq_length, 1, streamId);

        input_ids = input_ids_buf;
        input_mask = input_mask_buf;
        segment_ids = segment_ids_buf;
#endif

        // pre-compute buffers
        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            if (additional_output_layer != nullptr) {
                additional_output_layer->_pre_compute(batch_size, _logits);
            }
            buffer_filled = true;
        }

        // bert/embeddings
        bert_embeddings->compute(batch_size, input_ids, segment_ids, _embedding_output);

        // bert/encoder
        _sequence_output = transformer->_in_compute(batch_size, _embedding_output, input_mask);

        // bert/pooler
        bert_pooler->compute(batch_size, _sequence_output, _pooled_output);

        if (additional_output_layer != nullptr) {
            additional_output_layer->_in_compute(batch_size, _pooled_output, _logits, _probs);
        }

        // buffers should be re-computed in the next request
        buffer_filled = false;
    }

    template <typename T>
    void Bert<T>::logits(size_t batch_size, T *logits, T *probs) {
        if (additional_output_layer == nullptr) {
            std::cerr << "model does not have additional_output_layer, the output logits is wrong." << std::endl;
        }

        void *streamId = cuBERT::blas_get_stream(cublas);
        if (logits != nullptr) {
            cuBERT::memcpyAsync(logits, _logits, sizeof(T) * batch_size * num_labels, 2, streamId);
        }
        if (probs != nullptr) {
            cuBERT::memcpyAsync(probs, _probs, sizeof(T) * batch_size * num_labels, 2, streamId);
        }
        cuBERT::cuda_stream_synchronize(streamId);

        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            if (additional_output_layer != nullptr) {
                additional_output_layer->_pre_compute(batch_size, _logits);
            }
            buffer_filled = true;
        }
    }

    template <typename T>
    void Bert<T>::pooled_output(size_t batch_size, T *pooled_output) {
        void *streamId = cuBERT::blas_get_stream(cublas);
        cuBERT::memcpyAsync(pooled_output, _pooled_output, sizeof(T) * batch_size * hidden_size, 2, streamId);
        cuBERT::cuda_stream_synchronize(streamId);

        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            if (additional_output_layer != nullptr) {
                additional_output_layer->_pre_compute(batch_size, _logits);
            }
            buffer_filled = true;
        }
    }

    template <typename T>
    void Bert<T>::sequence_output(size_t batch_size, T *sequence_output) {
        void *streamId = cuBERT::blas_get_stream(cublas);
        cuBERT::memcpyAsync(sequence_output, _sequence_output,
                            sizeof(T) * batch_size * seq_length * hidden_size, 2, streamId);
        cuBERT::cuda_stream_synchronize(streamId);

        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            if (additional_output_layer != nullptr) {
                additional_output_layer->_pre_compute(batch_size, _logits);
            }
            buffer_filled = true;
        }
    }

    template <typename T>
    void Bert<T>::embedding_output(size_t batch_size, T *embedding_output) {
        void *streamId = cuBERT::blas_get_stream(cublas);
        cuBERT::memcpyAsync(embedding_output, _embedding_output,
                           sizeof(T) * batch_size * seq_length * hidden_size, 2, streamId);
        cuBERT::cuda_stream_synchronize(streamId);

        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            if (additional_output_layer != nullptr) {
                additional_output_layer->_pre_compute(batch_size, _logits);
            }
            buffer_filled = true;
        }
    }

    template class Bert<float>;
#ifdef HAVE_CUDA
    template class Bert<half>;
#endif
}
