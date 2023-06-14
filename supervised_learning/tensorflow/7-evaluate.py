#!/usr/bin/env python3
"""evaluate"""
import tensorflow as tf


def evaluate(X, Y, save_path):
"""evaluate func"""
    saver = tf.train.import_meta_graph(save_path + '.meta')

    # Get the tensors from the graph's collection
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('input_placeholder:0')
    y = graph.get_tensor_by_name('labels_placeholder:0')
    y_pred = graph.get_tensor_by_name('output_layer/y_pred:0')
    loss = graph.get_tensor_by_name('loss/loss:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

    # Start a TensorFlow session
    with tf.Session() as sess:
        # Restore the saved model
        saver.restore(sess, save_path)

        # Evaluate the model on the input data
        pred, acc, l = sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y})

    return pred, acc, l
