#include "cuBERT.h"
#include "cuBERT/common.h"
#include "cuBERT/multi/BertM.h"

void cuBERT_initialize(bool force_cpu) {
    cuBERT::initialize(force_cpu);
}

void cuBERT_finalize() {
    cuBERT::finalize();
}

void *cuBERT_open(const char *model_file,
                  int max_batch_size,
                  int seq_length,
                  int num_hidden_layers,
                  int num_attention_heads) {
    auto *model = new cuBERT::BertM(model_file,
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
    ((cuBERT::BertM *) model)->compute_cpu(batch_size, input_ids, input_mask, segment_ids, logits);
}

void cuBERT_close(void *model) {
    delete (cuBERT::BertM *) model;
}
