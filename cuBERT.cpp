#include "cuBERT.h"
#include "cuBERT/tf/Graph.h"
#include "cuBERT/bert/Bert.h"
#include "cuBERT/mgpu/BertMGPU.h"

void *cuBERT_open(const char *model_file,
                  int max_batch_size,
                  int seq_length,
                  int num_hidden_layers,
                  int num_attention_heads) {
    auto *model = new cuBERT::BertMGPU(model_file,
                                       max_batch_size,
                                       seq_length,
                                       num_hidden_layers, num_attention_heads);
    return model;
}

void cuBERT_compute(void *model,
                    int batch_size,
                    int *input_ids,
                    char *input_mask,
                    char *segment_ids,
                    float *logits) {
    ((cuBERT::BertMGPU *) model)->compute_cpu(batch_size, input_ids, input_mask, segment_ids, logits);
}

void cuBERT_close(void *model) {
    delete (cuBERT::BertMGPU *) model;
}

void* mklBERT_open(const char* model_file,
                   int max_batch_size,
                   int seq_length,
                   int num_hidden_layers,
                   int num_attention_heads) {
    cuBERT::Graph graph(model_file);

    auto *bert = new cuBERT::Bert(graph.var, max_batch_size, seq_length,
                                  graph.vocab_size,
                                  graph.type_vocab_size,
                                  graph.hidden_size,
                                  num_hidden_layers,
                                  num_attention_heads,
                                  graph.intermediate_size);
    return bert;
}

float *mklBERT_compute(void *model,
                       int batch_size,
                       int *input_ids,
                       char *input_mask,
                       char *segment_ids) {
    ((cuBERT::Bert *) model)->compute_cpu(batch_size, input_ids, input_mask, segment_ids);
    return ((cuBERT::Bert *) model)->get_logits_cpu();
}

void mklBERT_close(void* model) {
    delete (cuBERT::Bert *) model;
}
