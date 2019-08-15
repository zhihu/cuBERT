#ifndef CUBERT_BERT_H
#define CUBERT_BERT_H

#include <string>
#include <unordered_map>

#include "cuBERT.h"
#include "cuBERT/op_bert/BertEmbeddings.h"
#include "cuBERT/op_att/Transformer.h"
#include "cuBERT/op_bert/BertPooler.h"
#include "cuBERT/op_out/AdditionalOutputLayer.h"

namespace cuBERT {
    template <typename T>
    class Bert {
    public:
        explicit Bert(const std::unordered_map<std::string, T *> &var,
                      size_t max_batch_size,
                      size_t seq_length,
                      size_t vocab_size,
                      size_t type_vocab_size,
                      size_t hidden_size = 768,
                      size_t num_hidden_layers = 12,
                      size_t num_attention_heads = 12,
                      size_t intermediate_size = 3072,
                      size_t num_labels = 1);

        virtual ~Bert();

        // pre-compute buffers
        void _pre_compute(size_t batch_size);

        void compute(size_t batch_size, int *input_ids, int8_t *input_mask, int8_t *segment_ids);

        // ouput methods, cpu/gpu outputs
        void logits(size_t batch_size, T *logits, T *probs);
        void pooled_output(size_t batch_size, T *pooled_output);
        void sequence_output(size_t batch_size, T *sequence_output);
        void embedding_output(size_t batch_size, T *embedding_output);

        void output(size_t batch_size, cuBERT_Output* output);

    private:
        void* cublas;
        void* stream;

        size_t max_batch_size;
        size_t seq_length;
        size_t hidden_size;
        size_t num_labels;

        BertEmbeddings<T> *bert_embeddings;
        Transformer<T> *transformer;
        Pooler<T> *bert_pooler;
        ClassifierOutputLayer<T> *additional_output_layer;

        // input buffer
        int *input_ids_buf;
        int8_t *input_mask_buf;
        int8_t *segment_ids_buf;

        // cpu/gpu output buffers
        T *_embedding_output;
        T *_sequence_output;
        T *_pooled_output;
        T *_logits;
        T *_probs;

        // for pre-compute
        // FIXME: _sequence_output will be flushed
        bool buffer_filled;
    };
}

#endif //CUBERT_BERT_H
