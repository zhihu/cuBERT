import numpy as np
import tensorflow as tf

from bert import attention_layer, layer_norm

batch_size = 2
num_attention_heads = 2
size_per_head = 3
seq_length = 4
intermediate_size = 5

query_kernel = np.array([
    [-0.07848196, -0.18097023, 0.06933199, -0.07760319, 0.11389876, 0.05236414],
    [-0.02015782, 0.00233333, -0.00281469, -0.01525305, 0.17362033, -0.01600084],
    [0.00521428, 0.06063714, -0.10533229, 0.0228875, -0.00108843, -0.05974746],
    [-0.05530503, 0.06056419, 0.099603, 0.04929306, 0.08636444, 0.08424559],
    [0.02739674, -0.08676406, -0.0819858, 0.03834791, -0.03903558, 0.01903536],
    [0.01325864, 0.07587593, 0.20709228, -0.0421985, -0.10500058, -0.08004139]
], dtype=np.float32)
query_bias = np.array([-0.01566293, -0.01429354, -0.02946532, 0.02332242, -0.03551506, 0.00519018], dtype=np.float32)

key_kernel = np.array([
    [-0.19046976, -0.052291, 0.00774184, -0.04793982, -0.03272828, -0.07022775],
    [0.05397043, 0.22157724, -0.28796428, -0.13628182, 0.10769557, -0.04396444],
    [0.11023977, 0.11277004, -0.17019109, -0.00998783, -0.13503011, 0.03862515],
    [-0.00570178, -0.03683843, -0.09878516, -0.08536254, -0.20706373, 0.07736684],
    [0.09753255, 0.08549864, 0.07573727, -0.08459285, 0.11262332, -0.06660723],
    [-0.05978908, 0.04687774, 0.20048976, -0.15552515, -0.09287686, -0.05736409]
], dtype=np.float32)
key_bias = np.array([0.01119683, -0.00749641, 0.00929781, -0.00789247, 0.00374282, -0.0203852], dtype=np.float32)

value_kernel = np.array([
    [0.18298741, 0.13052676, 0.13003705, -0.07762788, -0.11298412, -0.09672086],
    [-0.27567647, -0.11159269, -0.20191047, -0.04961415, 0.03338585, -0.00217377],
    [0.0080993, -0.0092568, -0.07923323, -0.09595821, -0.0724212, 0.00234286],
    [0.08350474, 0.10685625, -0.03265393, 0.12026393, 0.11865459, 0.03879681],
    [0.09247954, -0.08354547, -0.04044447, 0.05576184, 0.063286, -0.06426957],
    [0.11189654, 0.04743394, 0.04952021, 0.06824017, -0.0718908, 0.06118326]
], dtype=np.float32)
value_bias = np.array([-0.01532887, -0.02567805, 0.02993296, 0.00255634, 0.03075514, -0.02086536], dtype=np.float32)

attention_output_kernel = np.array([
    [-0.02547911, 0.04877987, 0.05000711, 0.04084699, -0.08732582, 0.09071281],
    [-0.04081769, 0.21188675, 0.05063592, 0.04011015, -0.09087955, -0.02277032],
    [0.11330121, -0.00220912, -0.21545858, -0.0109133, 0.12117786, -0.07627827],
    [0.03476971, 0.113976, 0.0352498, -0.00169246, 0.17134688, 0.05991947],
    [-0.04367283, -0.08021438, 0.07809242, 0.04896554, -0.09109284, -0.17430527],
    [0.07785448, -0.08642721, 0.0911883, -0.00432356, -0.10407569, 0.03155923]
], dtype=np.float32)
attention_output_bias = np.array([0.00502381, 0.00164522, 0.00503161, 0.05414474, 0.00594567, -0.00136505],
                                 dtype=np.float32)

attention_norm_beta = np.array([0.01438165, 0.00893282, -0.00166658, -0.01515444, 0.01131669, -0.00312567],
                               dtype=np.float32)
attention_norm_gamma = np.array([0.98945833, 1.00672382, 1.00227484, 0.98692834, 1.00251162, 0.99780415],
                                dtype=np.float32)

intermediate_kernel = np.array([
    [0.04110776, 0.00867842, -0.11692518, 0.00942204, 0.00212334],
    [0.03458865, -0.00608362, -0.12785568, 0.01738149, -0.0735809],
    [-0.03358123, -0.02204291, 0.19460295, 0.10060768, -0.11971488],
    [0.02828389, -0.07767208, 0.03127521, 0.01363018, -0.14119004],
    [0.01852505, -0.12854275, 0.0481119, -0.15679542, -0.08593457],
    [0.00225799, -0.03674033, -0.10633834, 0.03639213, 0.07383945]
], dtype=np.float32)
intermediate_bias = np.array([-0.0094941, 0.00329734, 0.00365913, 0.02430543, 0.04413794], dtype=np.float32)

output_kernel = np.array([
    [0.14096574, 0.0019019, 0.03194073, -0.01783772, 0.04542776, -0.17121975],
    [-0.03054714, -0.03382285, -0.14785342, -0.04588855, -0.09048948, -0.04335051],
    [0.12839685, -0.17706056, -0.01360187, 0.02532171, 0.08845975, 0.00350385],
    [0.07184936, 0.11032352, 0.0339272, -0.04756412, -0.20521204, 0.12666636],
    [0.06397831, -0.15246845, -0.00572673, -0.09259837, -0.00063671, -0.13432225]
], dtype=np.float32)
output_bias = np.array([-0.01755394, 0.02878171, 0.04216052, 0.01562296, 0.01129209, 0.04988396], dtype=np.float32)

output_norm_beta = np.array([0.00422856, 0.04091637, 0.03255221, -0.03470522, 0.01916321, -0.00184435],
                            dtype=np.float32)
output_norm_gamma = np.array([0.9903053, 0.95159506, 0.98762059, 0.99406842, 1.00686035, 0.97648946],
                             dtype=np.float32)


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def transformer_model(input_tensor,
                      neg_attention_mask,
                      num_hidden_layers=12,
                      intermediate_act_fn=gelu):
    hidden_size = num_attention_heads * size_per_head

    neg_attention_mask = tf.reshape(neg_attention_mask, [batch_size, 1, 1, seq_length])
    neg_attention_mask *= tf.ones(shape=[batch_size, num_attention_heads, seq_length, seq_length],
                                  dtype=tf.float32)

    prev_output = input_tensor

    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        layer_input, neg_attention_mask,
                        query_kernel=query_kernel, query_bias=query_bias,
                        key_kernel=key_kernel, key_bias=key_bias,
                        value_kernel=value_kernel, value_bias=value_bias,
                        num_attention_heads=num_attention_heads,
                        size_per_head=size_per_head,
                        batch_size=batch_size,
                        seq_length=seq_length)

                with tf.variable_scope("output"):
                    attention_output = tf.layers.Dense(
                        hidden_size,
                        weights=[attention_output_kernel, attention_output_bias]
                    ).apply(attention_head)
                    attention_output = layer_norm(attention_output + layer_input,
                                                  beta=attention_norm_beta,
                                                  gamma=attention_norm_gamma)

            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.Dense(
                    intermediate_size,
                    activation=intermediate_act_fn,
                    weights=[intermediate_kernel, intermediate_bias]
                ).apply(attention_output)

            with tf.variable_scope("output"):
                layer_output = tf.layers.Dense(
                    hidden_size,
                    weights=[output_kernel, output_bias]
                ).apply(intermediate_output)
                layer_output = layer_norm(layer_output + attention_output,
                                          beta=output_norm_beta,
                                          gamma=output_norm_gamma)
                prev_output = layer_output

    return prev_output


if __name__ == '__main__':
    tensor = tf.placeholder(tf.float32, shape=[batch_size * seq_length, num_attention_heads * size_per_head])
    neg_attention_mask = tf.placeholder(tf.float32, shape=[batch_size, seq_length])

    tensor_ = np.arange(48, dtype=np.float32)
    tensor_ = np.reshape(tensor_, (8, 6))

    neg_attention_mask_ = np.zeros((batch_size, seq_length), dtype=np.float32)
    neg_attention_mask_[0, 0] = 1

    with tf.Session() as sess:
        output = transformer_model(tensor, neg_attention_mask, num_hidden_layers=2)
        output_ = sess.run(output, feed_dict={
            tensor: tensor_,
            neg_attention_mask: neg_attention_mask_,
        })
        print output_
