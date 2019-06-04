cdef extern from "../src/cuBERT.h":
    cdef enum cuBERT_ComputeType:
        cuBERT_COMPUTE_FLOAT, cuBERT_COMPUTE_HALF
    
    cdef enum cuBERT_OutputType:
        cuBERT_LOGITS, cuBERT_POOLED_OUTPUT, cuBERT_SEQUENCE_OUTPUT, cuBERT_EMBEDDING_OUTPUT, cuBERT_PROBS

    void cuBERT_initialize() except +;
    void cuBERT_finalize() except +;

    void* cuBERT_open(const char* model_file,
                      int max_batch_size,
                      int seq_length,
                      int num_hidden_layers,
                      int num_attention_heads,
                      cuBERT_ComputeType compute_type) except +;

    void cuBERT_compute(void* model,
                        int batch_size,
                        int* input_ids,
                        signed char* input_mask,
                        signed char* segment_ids,
                        void* output,
                        cuBERT_OutputType output_type,
                        cuBERT_ComputeType compute_type) except +;

    void cuBERT_close(void* model,
                      cuBERT_ComputeType compute_type) except +;

    void* cuBERT_open_tokenizer(const char* vocab_file, int do_lower_case) except +;

    void cuBERT_close_tokenizer(void* tokenizer) except +;

    void cuBERT_tokenize_compute(void* model,
                                 void* tokenizer,
                                 int batch_size,
                                 const char** text_a,
                                 const char** text_b,
                                 void* output,
                                 cuBERT_OutputType output_type,
                                 cuBERT_ComputeType compute_type) except +;
