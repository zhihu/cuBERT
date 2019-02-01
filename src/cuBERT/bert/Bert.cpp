#include "cuBERT/common.h"
#include "Bert.h"

namespace cuBERT {
    Bert::Bert(const std::unordered_map<std::string, float *> &var,
               size_t max_batch_size,
               size_t seq_length,
               size_t vocab_size,
               size_t type_vocab_size,
               size_t hidden_size,
               size_t num_hidden_layers,
               size_t num_attention_heads,
               size_t intermediate_size) {
        this->seq_length = seq_length;
        this->hidden_size = hidden_size;

        this->stream = cuBERT::cuda_stream_create();
        this->cublas = cuBERT::blas_create();
        cuBERT::blas_set_stream(cublas, stream);

        this->bert_embeddings = new BertEmbeddings(cublas, var, max_batch_size,
                                                   vocab_size, type_vocab_size, hidden_size, seq_length);

        this->transformer = new Transformer(cublas, "bert/encoder", var,
                                            max_batch_size, seq_length,
                                            hidden_size, num_hidden_layers, num_attention_heads, intermediate_size);

        this->bert_pooler = new BertPooler(cublas, seq_length, hidden_size,
                                           var.at("bert/pooler/dense/kernel"),
                                           var.at("bert/pooler/dense/bias"),
                                           max_batch_size);

        this->additional_output_layer = new AdditionalOutputLayer(cublas, hidden_size, var.at("output_weights"));

        this->_embedding_output = static_cast<float *>(cuBERT::malloc(sizeof(float) * max_batch_size * seq_length * hidden_size));
        this->_pooled_output = static_cast<float *>(cuBERT::malloc(sizeof(float) * max_batch_size * hidden_size));
        this->_logits = static_cast<float *>(cuBERT::malloc(sizeof(float) * max_batch_size));

        this->input_ids_buf = static_cast<int *>(cuBERT::malloc(sizeof(int) * max_batch_size * seq_length));
        this->input_mask_buf = static_cast<char *>(cuBERT::malloc(sizeof(char) * max_batch_size * seq_length));
        this->segment_ids_buf = static_cast<char *>(cuBERT::malloc(sizeof(char) * max_batch_size * seq_length));

        // pre-compute buffers
        transformer->_pre_compute(max_batch_size);
        this->buffer_filled = true;
    }

    Bert::~Bert() {
        cuBERT::free(segment_ids_buf);
        cuBERT::free(input_mask_buf);
        cuBERT::free(input_ids_buf);

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

    void Bert::compute(size_t batch_size, int *input_ids, char *input_mask, char *segment_ids) {
        if (cuBERT::gpu()) {
            // copy inputs
            void *streamId = blas_get_stream(cublas);
            cuBERT::memcpyAsync(input_ids_buf, input_ids, sizeof(int) * batch_size * seq_length, 1, streamId);
            cuBERT::memcpyAsync(input_mask_buf, input_mask, sizeof(char) * batch_size * seq_length, 1, streamId);
            cuBERT::memcpyAsync(segment_ids_buf, segment_ids, sizeof(char) * batch_size * seq_length, 1, streamId);

            input_ids = input_ids_buf;
            input_mask = input_mask_buf;
            segment_ids = segment_ids_buf;
        }

        // pre-compute buffers
        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            buffer_filled = true;
        }

        // bert/embeddings
        bert_embeddings->compute(batch_size, input_ids, segment_ids, _embedding_output);

        // bert/encoder
        _sequence_output = transformer->_in_compute(batch_size, _embedding_output, input_mask);

        // bert/pooler
        bert_pooler->compute(batch_size, _sequence_output, _pooled_output);

        additional_output_layer->compute(batch_size, _pooled_output, _logits);

        // buffers should be re-computed in the next request
        buffer_filled = false;
    }

    void Bert::logits(size_t batch_size, float *logits) {
        void *streamId = cuBERT::blas_get_stream(cublas);
        cuBERT::memcpyAsync(logits, _logits, sizeof(float) * batch_size, 2, streamId);
        cuBERT::cuda_stream_synchronize(streamId);

        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            buffer_filled = true;
        }
    }

    void Bert::pooled_output(size_t batch_size, float *pooled_output) {
        void *streamId = cuBERT::blas_get_stream(cublas);
        cuBERT::memcpyAsync(pooled_output, _pooled_output, sizeof(float) * batch_size * hidden_size, 2, streamId);
        cuBERT::cuda_stream_synchronize(streamId);

        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            buffer_filled = true;
        }
    }

    void Bert::sequence_output(size_t batch_size, float *sequence_output) {
        void *streamId = cuBERT::blas_get_stream(cublas);
        cuBERT::memcpyAsync(sequence_output, _sequence_output,
                            sizeof(float) * batch_size * seq_length * hidden_size, 2, streamId);
        cuBERT::cuda_stream_synchronize(streamId);

        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            buffer_filled = true;
        }
    }

    void Bert::embedding_output(size_t batch_size, float *embedding_output) {
        void *streamId = cuBERT::blas_get_stream(cublas);
        cuBERT::memcpyAsync(embedding_output, _embedding_output,
                           sizeof(float) * batch_size * seq_length * hidden_size, 2, streamId);
        cuBERT::cuda_stream_synchronize(streamId);

        if (!buffer_filled) {
            transformer->_pre_compute(batch_size);
            buffer_filled = true;
        }
    }

    float *Bert::get_logits() {
        return this->_logits;
    }

    float *Bert::get_embedding_output() {
        return this->_embedding_output;
    }
}
