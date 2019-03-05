#ifndef CUBERT_BERT_H
#define CUBERT_BERT_H

#include <string>
#include <unordered_map>

#include "cuBERT/op_bert/BertEmbeddings.h"
#include "cuBERT/op_att/Transformer.h"
#include "cuBERT/op_bert/BertPooler.h"
#include "cuBERT/op_out/AdditionalOutputLayer.h"

namespace cuBERT {
    class Bert {
    public:
        explicit Bert(const std::unordered_map<std::string, float *> &var,
                      size_t max_batch_size,
                      size_t seq_length,
                      size_t vocab_size,
                      size_t type_vocab_size,
                      size_t hidden_size = 768,
                      size_t num_hidden_layers = 12,
                      size_t num_attention_heads = 12,
                      size_t intermediate_size = 3072);

        virtual ~Bert();

        void compute(size_t batch_size, int *input_ids, char *input_mask, char *segment_ids);

        // ouput methods, cpu/gpu outputs
        void logits(size_t batch_size, float *logits);
        void pooled_output(size_t batch_size, float *pooled_output);
        void sequence_output(size_t batch_size, float *sequence_output);
        void embedding_output(size_t batch_size, float *embedding_output);

    private:
        void* cublas;
        void* stream;

        size_t max_batch_size;
        size_t seq_length;
        size_t hidden_size;

        BertEmbeddings *bert_embeddings;
        Transformer *transformer;
        Pooler *bert_pooler;
        AdditionalOutputLayer *additional_output_layer;

        // input buffer
        int *input_ids_buf;
        char *input_mask_buf;
        char *segment_ids_buf;

        // cpu/gpu output buffers
        float *_embedding_output;
        float *_sequence_output;
        float *_pooled_output;
        float *_logits;

        // for pre-compute
        // FIXME: _sequence_output will be flushed
        bool buffer_filled;
    };
}

#endif //CUBERT_BERT_H
