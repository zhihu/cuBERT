package com.zhihu.cubert;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

interface CLibrary extends Library {
    CLibrary INSTANCE = Native.load("cuBERT", CLibrary.class);

    void cuBERT_initialize();

    void cuBERT_finalize();

    int cuBERT_get_gpu_count();

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

    @Structure.FieldOrder({"logits", "pooled_output", "sequence_output", "embedding_output", "probs"})
    class cuBERT_Output extends Structure {
        public ByteBuffer logits;
        public ByteBuffer pooled_output;
        public ByteBuffer sequence_output;
        public ByteBuffer embedding_output;
        public ByteBuffer probs;

        cuBERT_Output(Output output) {
            this.logits = allocate(output.logits);
            this.pooled_output = allocate(output.pooledOutput);
            this.sequence_output = allocate(output.sequenceOutput);
            this.embedding_output = allocate(output.embeddingOutput);
            this.probs = allocate(output.probs);
        }

        private static ByteBuffer allocate(float[] array) {
            if (array == null) {
                return null;
            }
            int outputSize = array.length;
            int elementSize = 4;
            return ByteBuffer.allocateDirect(outputSize * elementSize).order(ByteOrder.nativeOrder());
        }
    }

    void cuBERT_compute_m(Pointer model,
                          int batch_size,
                          int[] input_ids,
                          byte[] input_mask,
                          byte[] segment_ids,
                          cuBERT_Output output,
                          int compute_type,
                          int output_to_float);

    void cuBERT_tokenize_compute_m(Pointer model,
                                   Pointer tokenizer,
                                   int batch_size,
                                   String[] text_a,
                                   String[] text_b,
                                   cuBERT_Output output,
                                   int compute_type,
                                   int output_to_float);
}
