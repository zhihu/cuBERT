import numpy as np
import tensorflow as tf

from bert import attention_layer

batch_size = 5
num_attention_heads = 1
size_per_head = 3
seq_length = 2

query_kernel = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
], dtype=np.float32)
query_bias = np.array([-3, -2, -1], dtype=np.float32)

key_kernel = np.array([
    [8, 7, 6],
    [5, 4, 3],
    [2, 1, 0]
], dtype=np.float32)
key_bias = np.array([0, 1, 2], dtype=np.float32)

value_kernel = np.array([
    [-1, -1, -1],
    [2, 2, 2],
    [1, 1, 1]
], dtype=np.float32)
value_bias = np.array([3, 3, 3], dtype=np.float32)


if __name__ == '__main__':
    tensor = tf.placeholder(tf.float32, shape=[None, seq_length, num_attention_heads * size_per_head])
    neg_attention_mask = tf.placeholder(tf.float32, shape=[None, num_attention_heads, seq_length, seq_length])

    tensor_ = np.arange(30, dtype=np.float32)
    tensor_ = np.reshape(tensor_, (batch_size, seq_length, num_attention_heads * size_per_head))

    neg_attention_mask_ = np.zeros((batch_size, num_attention_heads, seq_length, seq_length), dtype=np.float32)
    neg_attention_mask_[0, 0, 0, 0] = 1

    with tf.Session() as sess:
        context_layer = attention_layer(tensor, neg_attention_mask,
                                        query_kernel=query_kernel, query_bias=query_bias,
                                        key_kernel=key_kernel, key_bias=key_bias,
                                        value_kernel=value_kernel, value_bias=value_bias,
                                        num_attention_heads=num_attention_heads,
                                        size_per_head=size_per_head,
                                        batch_size=batch_size,
                                        seq_length=seq_length)

        context_layer_ = sess.run(context_layer, feed_dict={
            tensor: tensor_,
            neg_attention_mask: neg_attention_mask_,
        })
        print context_layer_
