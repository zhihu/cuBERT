#ifndef CUBERT_LIBRARY_H
#define CUBERT_LIBRARY_H

#if __cplusplus <= 199711L
typedef signed char int8_t;
#else
#include <cstdint>
#endif

extern "C" {

enum cuBERT_ComputeType {
    cuBERT_COMPUTE_FLOAT = 0,
    cuBERT_COMPUTE_HALF = 1, /** half precision */
};

enum cuBERT_OutputType {
    cuBERT_LOGITS = 0,
    cuBERT_POOLED_OUTPUT = 1,
    cuBERT_SEQUENCE_OUTPUT = 2,
    cuBERT_EMBEDDING_OUTPUT = 3,
};

void cuBERT_initialize();
void cuBERT_finalize();

void* cuBERT_open(const char* model_file,
                  int max_batch_size,
                  int seq_length,
                  int num_hidden_layers,
                  int num_attention_heads,
                  cuBERT_ComputeType compute_type = cuBERT_COMPUTE_FLOAT);

void cuBERT_compute(void* model,
                    int batch_size,
                    int* input_ids,
                    int8_t* input_mask,
                    int8_t* segment_ids,
                    void* output,
                    cuBERT_OutputType output_type = cuBERT_LOGITS,
                    cuBERT_ComputeType compute_type = cuBERT_COMPUTE_FLOAT);

void cuBERT_close(void* model,
                  cuBERT_ComputeType compute_type = cuBERT_COMPUTE_FLOAT);

/** high level API including tokenization **/

void* cuBERT_open_tokenizer(const char* vocab_file, int do_lower_case = 1);

void cuBERT_close_tokenizer(void* tokenizer);

void cuBERT_tokenize_compute(void* model,
                             void* tokenizer,
                             int batch_size,
                             const char** text_a,
                             const char** text_b,
                             void* output,
                             cuBERT_OutputType output_type = cuBERT_LOGITS,
                             cuBERT_ComputeType compute_type = cuBERT_COMPUTE_FLOAT);

}

#endif