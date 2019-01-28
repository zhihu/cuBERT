#ifndef CUBERT_BERT_H
#define CUBERT_BERT_H


#include <cublas_v2.h>
#include <cudnn.h>

#include "./BertEmbeddings.h"
#include "./Transformer.h"
#include "./BertPooler.h"
#include "./AdditionalOutputLayer.h"

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

        void compute_cpu(size_t batch_size, int *input_ids, char *input_mask, char *segment_ids);

        void logits(size_t batch_size, float *logits);

        void embedding_output(size_t batch_size, float *embedding_output);

    private:
        cublasHandle_t cublas;
        cudnnHandle_t cudnn;
        cudaStream_t stream;

        size_t seq_length;
        size_t hidden_size;

        BertEmbeddings *bert_embeddings;
        Transformer *transformer;
        BertPooler *bert_pooler;
        AdditionalOutputLayer *additional_output_layer;

        // buffer
        int *input_ids_gpu;
        char *input_mask_gpu;
        char *segment_ids_gpu;

        float *embedding_output_gpu;
        float *sequence_output_gpu;
        float *pooled_output_gpu;

        float *logits_gpu;

        // for pre-compute
        // FIXME: sequence_output_gpu will be flushed
        bool buffer_filled;
    };
}

#endif //CUBERT_BERT_H
