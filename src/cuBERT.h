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
    cuBERT_PROBS = 4,
};

void cuBERT_initialize();
void cuBERT_finalize();

int cuBERT_get_gpu_count();

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

/** new API with multiple outputs **/

struct cuBERT_Output {
    void* logits = nullptr;
    void* pooled_output = nullptr;
    void* sequence_output = nullptr;
    void* embedding_output = nullptr;
    void* probs = nullptr;
};

// output_to_float = true:
//     for half model, the output should be always float, the method will convert half to float internally;
//     for float model, this flag is not used.
// output_to_float = false:
//     the output data type should be equal to compute_type.
void cuBERT_compute_m(void* model,
                      int batch_size,
                      int* input_ids,
                      int8_t* input_mask,
                      int8_t* segment_ids,
                      cuBERT_Output* output,
                      cuBERT_ComputeType compute_type = cuBERT_COMPUTE_FLOAT,
                      int output_to_float = 0);

void cuBERT_tokenize_compute_m(void* model,
                               void* tokenizer,
                               int batch_size,
                               const char** text_a,
                               const char** text_b,
                               cuBERT_Output* output,
                               cuBERT_ComputeType compute_type = cuBERT_COMPUTE_FLOAT,
                               int output_to_float = 0);

}

#endif