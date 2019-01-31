#ifndef CUBERT_LIBRARY_H
#define CUBERT_LIBRARY_H

extern "C" {

void cuBERT_initialize(bool force_cpu = false);
void cuBERT_finalize();

void* cuBERT_open(const char* model_file,
                  int max_batch_size,
                  int seq_length,
                  int num_hidden_layers,
                  int num_attention_heads);

void cuBERT_compute(void* model,
                    int batch_size,
                    int* input_ids,
                    char* input_mask,
                    char* segment_ids,
                    float* logits);

void cuBERT_close(void* model);

}

#endif