#include "cuBERT.h"
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
