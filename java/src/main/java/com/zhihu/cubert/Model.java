package com.zhihu.cubert;

import com.sun.jna.Pointer;

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
        Output o = new Output(output, outputType);
        compute(batchSize, inputIds, inputMask, segmentIds, o);
    }

    public void compute(int batchSize,
                        int[] inputIds,
                        byte[] inputMask,
                        byte[] segmentIds,
                        Output output) {
        CLibrary.cuBERT_Output c_output = new CLibrary.cuBERT_Output(output, computeType);
        CLibrary.INSTANCE.cuBERT_compute_m(
                _c_model, batchSize, inputIds, inputMask, segmentIds, c_output, computeType.ordinal());
        output.fillOutput(c_output, computeType);
    }

    public void tokenizeCompute(String[] textA, String[] textB, Number[] output, OutputType outputType) {
        Output o = new Output(output, outputType);
        tokenizeCompute(textA, textB, o);
    }

    public void tokenizeCompute(String[] textA, String[] textB, Output output) {
        int batchSize = textA.length;
        if (textB != null && batchSize != textB.length) {
            throw new IllegalArgumentException("batch_size mismatch");
        }

        CLibrary.cuBERT_Output c_output = new CLibrary.cuBERT_Output(output, computeType);
        CLibrary.INSTANCE.cuBERT_tokenize_compute_m(
                _c_model, _c_tokenizer, batchSize, textA, textB, c_output, computeType.ordinal());
        output.fillOutput(c_output, computeType);
    }
}
