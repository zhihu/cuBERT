import numpy as np
import tensorflow as tf

frozen_graph_filename = '../bert_frozen_seq32.pb'

input_ids_ = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
], dtype=np.int64)

input_mask_ = np.array([
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]
], dtype=np.int64)

segment_ids_ = np.array([
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
], dtype=np.int64)

with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.graph_util.import_graph_def(graph_def)

    for op in graph.get_operations():
        print(op.name, op.values())

    input_ids = graph.get_tensor_by_name('import/input_ids:0')
    input_mask = graph.get_tensor_by_name('import/input_mask:0')
    segment_ids = graph.get_tensor_by_name('import/segment_ids:0')
    output = graph.get_tensor_by_name('import/loss/output:0')

    embedding_output = graph.get_tensor_by_name('import/bert/embeddings/LayerNorm/batchnorm/add_1:0')

    with tf.Session(graph=graph) as sess:
        output_, embedding_output_ = sess.run((output, embedding_output), feed_dict={
            input_ids: input_ids_,
            input_mask: input_mask_,
            segment_ids: segment_ids_,
        })
        print output_
        print embedding_output_
