import math
import tensorflow as tf


def transpose_for_scores(input_tensor, _batch_size, _num_attention_heads, _seq_length, width):
    output_tensor = tf.reshape(input_tensor, [_batch_size, _seq_length, _num_attention_heads, width])
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor


def attention_layer(tensor,
                    neg_attention_mask,
                    query_kernel, query_bias,
                    key_kernel, key_bias,
                    value_kernel, value_bias,
                    num_attention_heads=1,
                    size_per_head=512,
                    batch_size=None,
                    seq_length=None):
    """
    :param seq_length:
    :param batch_size:
    :param size_per_head:
    :param num_attention_heads:
    :param value_bias:
    :param value_kernel:
    :param key_bias:
    :param key_kernel:
    :param query_bias:
    :param query_kernel:
    :param tensor: float Tensor of shape [batch_size, seq_length, width].
    :param neg_attention_mask: [batch_size, num_attention_heads, seq_length, seq_length]
    :return:
    """
    query_layer = tf.layers.Dense(
        num_attention_heads * size_per_head,
        name="query",
        weights=[query_kernel, query_bias]
    ).apply(tensor)

    key_layer = tf.layers.Dense(
        num_attention_heads * size_per_head,
        name="key",
        weights=[key_kernel, key_bias]
    ).apply(tensor)

    value_layer = tf.layers.Dense(
        num_attention_heads * size_per_head,
        name="value",
        weights=[value_kernel, value_bias]
    ).apply(tensor)

    query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, seq_length, size_per_head)
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads, seq_length, size_per_head)

    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    attention_scores += neg_attention_mask * -10000.0

    attention_probs = tf.nn.softmax(attention_scores)

    value_layer = tf.reshape(value_layer, [batch_size, seq_length, num_attention_heads, size_per_head])
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    context_layer = tf.matmul(attention_probs, value_layer)
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    context_layer = tf.reshape(context_layer, [batch_size * seq_length, num_attention_heads * size_per_head])
    return context_layer


def layer_norm(inputs,
               beta, gamma,
               begin_norm_axis=-1,
               begin_params_axis=-1):
        inputs_shape = inputs.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) must be < rank(inputs) (%d)' %
                             (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
                             (inputs.name, begin_params_axis, inputs_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta = tf.constant(beta, dtype=dtype)
        gamma = tf.constant(gamma, dtype=dtype)
        # Calculate the moments on the last axis (layer activations).
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        variance_epsilon = 1e-12
        outputs = tf.nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=variance_epsilon)
        outputs.set_shape(inputs_shape)
        return outputs
