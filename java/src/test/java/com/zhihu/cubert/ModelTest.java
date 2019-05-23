package com.zhihu.cubert;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class ModelTest {

    private Model model;

    @Before
    public void setUp() {
        model = new Model(
                "../build/bert_frozen_seq32.pb",
                128, 32, 12, 12,
                ComputeType.FLOAT,
                "../build/vocab.txt", 1);
    }

    @After
    public void tearDown() {
        model.close();
    }

    @Test
    public void testCompute() {
        int[] input_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

        byte[] input_mask = {1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0};

        byte[] segment_ids = {1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
                0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0};

        Float[] output = new Float[2];

        model.compute(2, input_ids, input_mask, segment_ids, output, OutputType.LOGITS);

        Assert.assertEquals(-2.9427543, output[0], 1e-5);
        Assert.assertEquals(-1.4876306, output[1], 1e-5);
    }

    @Test
    public void testTokenizeCompute() {
        String[] textA = new String[]{"知乎", "知乎"};
        String[] textB = new String[]{"在家刷知乎", "知乎发现更大的世界"};
        Float[] output = new Float[2];

        model.tokenizeCompute(textA, textB, output, OutputType.LOGITS);

        Assert.assertEquals(-2.51366, output[0], 1e-5);
        Assert.assertEquals(-1.47348, output[1], 1e-5);
    }
}
