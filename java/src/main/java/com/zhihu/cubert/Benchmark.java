package com.zhihu.cubert;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.profile.CompilerProfiler;
import org.openjdk.jmh.profile.HotspotCompilationProfiler;
import org.openjdk.jmh.profile.HotspotThreadProfiler;
import org.openjdk.jmh.profile.StackProfiler;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class Benchmark {

    private static final int max_batch_size = 128;
    private static final int batch_size = 128;
    private static final int seq_length = 32;
    private static final int hidden_size = 768;

    private static final int output_size = batch_size * hidden_size;

    private final int[] input_ids = new int[batch_size * seq_length];
    private final byte[] input_mask = new byte[batch_size * seq_length];
    private final byte[] segment_ids = new byte[batch_size * seq_length];
    private final float[] output = new float[output_size];

    private Model model;

    @Setup
    public void before() {
        Random random = new Random();
        for (int i = 0; i < batch_size * seq_length; i++) {
            input_ids[i] = random.nextInt(21120);
            input_mask[i] = (byte) random.nextInt(1);
            segment_ids[i] = (byte) random.nextInt(1);
        }

        model = new Model(
                "../build/bert_frozen_seq32.pb",
                max_batch_size, seq_length, 12, 12,
                ComputeType.FLOAT,
                "../build/vocab.txt", 1);
    }

    @TearDown
    public void after() {
        model.close();
    }

    @org.openjdk.jmh.annotations.Benchmark
    public void run() {
        model.compute(batch_size, input_ids, input_mask, segment_ids, output, OutputType.POOLED_OUTPUT);
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(Benchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(10)
                .measurementIterations(10)
                .addProfiler(StackProfiler.class)
                .addProfiler(HotspotThreadProfiler.class)
                .addProfiler(CompilerProfiler.class)
                .addProfiler(HotspotCompilationProfiler.class)
                .build();
        new Runner(opt).run();
    }
}
