package com.zhihu.cubert;

import java.nio.ByteBuffer;

public class Output {
    public float[] logits;
    public float[] pooledOutput;
    public float[] sequenceOutput;
    public float[] embeddingOutput;
    public float[] probs;

    public Output() {}

    public Output(float[] output, OutputType outputType) {
        switch (outputType) {
            case LOGITS:
                logits = output;
                break;
            case POOLED_OUTPUT:
                pooledOutput = output;
                break;
            case SEQUENCE_OUTPUT:
                sequenceOutput = output;
                break;
            case EMBEDDING_OUTPUT:
                embeddingOutput = output;
                break;
            case PROBS:
                probs = output;
                break;
            default:
                throw new IllegalArgumentException("unknown OutputType");
        }
    }

    void fillOutput(CLibrary.cuBERT_Output output) {
        fillOutput(output.logits, logits);
        fillOutput(output.pooled_output, pooledOutput);
        fillOutput(output.sequence_output, sequenceOutput);
        fillOutput(output.embedding_output, embeddingOutput);
        fillOutput(output.probs, probs);
    }

    private static void fillOutput(ByteBuffer buffer, float[] output) {
        if (buffer == null || output == null) {
            return;
        }
        buffer.asFloatBuffer().get(output);
    }
}
