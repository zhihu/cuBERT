package com.zhihu.cubert;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.nio.Buffer;

interface CLibrary extends Library {
    CLibrary INSTANCE = Native.load("cuBERT", CLibrary.class);

    void cuBERT_initialize();

    void cuBERT_finalize();

    Pointer cuBERT_open(String model_file,
                        int max_batch_size,
                        int seq_length,
                        int num_hidden_layers,
                        int num_attention_heads,
                        int compute_type);

    void cuBERT_compute(Pointer model,
                        int batch_size,
                        int[] input_ids,
                        byte[] input_mask,
                        byte[] segment_ids,
                        Buffer output,
                        int output_type,
                        int compute_type);

    void cuBERT_close(Pointer model,
                      int compute_type);

    Pointer cuBERT_open_tokenizer(String vocab_file, int do_lower_case);

    void cuBERT_close_tokenizer(Pointer tokenizer);

    void cuBERT_tokenize_compute(Pointer model,
                                 Pointer tokenizer,
                                 int batch_size,
                                 String[] text_a,
                                 String[] text_b,
                                 Buffer output,
                                 int output_type,
                                 int compute_type);
}
