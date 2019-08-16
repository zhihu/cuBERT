package com.zhihu.cubert;

import android.util.Half;

import java.nio.ByteBuffer;

public class Output {
    public Number[] logits;
    public Number[] pooledOutput;
    public Number[] sequenceOutput;
    public Number[] embeddingOutput;
    public Number[] probs;

    public Output() {}

    public Output(Number[] output, OutputType outputType) {
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

    void fillOutput(CLibrary.cuBERT_Output output, ComputeType computeType) {
        fillOutput(output.logits, logits, computeType);
        fillOutput(output.pooled_output, pooledOutput, computeType);
        fillOutput(output.sequence_output, sequenceOutput, computeType);
        fillOutput(output.embedding_output, embeddingOutput, computeType);
        fillOutput(output.probs, probs, computeType);
    }

    private static void fillOutput(ByteBuffer buffer, Number[] output, ComputeType computeType) {
        if (buffer == null || output == null) {
            return;
        }
        if (computeType == ComputeType.HALF) {
            for (int i = 0 ; i < output.length; i++) {
                output[i] = new Half(buffer.getShort());
            }
        } else {
            for (int i = 0; i < output.length; i++) {
                output[i] = buffer.getFloat();
            }
        }
    }
}
