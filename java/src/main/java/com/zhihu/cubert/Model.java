package com.zhihu.cubert;

import android.util.Half;
import com.sun.jna.Pointer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class Model implements AutoCloseable {

    static {
        CLibrary.INSTANCE.cuBERT_initialize();
    }

    private Pointer _c_model;

    private Pointer _c_tokenizer;

    private ComputeType computeType;

    public Model(String modelFile,
                 int maxBatchSize,
                 int seqLength,
                 int numHiddenLayers,
                 int numAttentionHeads,
                 ComputeType computeType,
                 String vocabFile,
                 int doLowerCase) {
        this._c_model = CLibrary.INSTANCE.cuBERT_open(
                modelFile, maxBatchSize, seqLength, numHiddenLayers, numAttentionHeads, computeType.ordinal());
        this.computeType = computeType;
        if (vocabFile != null) {
            this._c_tokenizer = CLibrary.INSTANCE.cuBERT_open_tokenizer(vocabFile, doLowerCase);
        }
    }

    @Override
    public void close() {
        CLibrary.INSTANCE.cuBERT_close(_c_model, computeType.ordinal());
        if (_c_tokenizer != null) {
            CLibrary.INSTANCE.cuBERT_close_tokenizer(_c_tokenizer);
        }
    }

    public void compute(int batchSize,
                        int[] inputIds,
                        byte[] inputMask,
                        byte[] segmentIds,
                        Number[] output,
                        OutputType outputType) {
        int outputSize = output.length;
        int elementSize = computeType == ComputeType.HALF ? 2 : 4;
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputSize * elementSize).order(ByteOrder.nativeOrder());

        CLibrary.INSTANCE.cuBERT_compute(
                _c_model, batchSize, inputIds, inputMask, segmentIds,
                outputBuffer, outputType.ordinal(), computeType.ordinal());

        fillOutput(outputBuffer, output);
    }

    public void tokenizeCompute(String[] textA, String[] textB, Number[] output, OutputType outputType) {
        int batchSize = textA.length;
        if (textB != null && batchSize != textB.length) {
            throw new IllegalArgumentException("batch_size mismatch");
        }

        int outputSize = output.length;
        int elementSize = computeType == ComputeType.HALF ? 2 : 4;
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputSize * elementSize).order(ByteOrder.nativeOrder());

        CLibrary.INSTANCE.cuBERT_tokenize_compute(
                _c_model, _c_tokenizer, batchSize, textA, textB,
                outputBuffer, outputType.ordinal(), computeType.ordinal());

        fillOutput(outputBuffer, output);
    }

    private void fillOutput(ByteBuffer buffer, Number[] output) {
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
