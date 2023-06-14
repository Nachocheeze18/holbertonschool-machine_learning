#!/usr/bin/env python3
"""train process"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    tf.reset_default_graph()

    X_train_tf = tf.constant(X_train, dtype=tf.float32)
    Y_train_tf = tf.constant(Y_train, dtype=tf.float32)
    X_valid_tf = tf.constant(X_valid, dtype=tf.float32)
    Y_valid_tf = tf.constant(Y_valid, dtype=tf.float32)

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    y_pred = forward_prop(x, layer_sizes, activations)

    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    train_op = create_train_op(loss, alpha)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            _, cost = sess.run([train_op, loss], feed_dict={x: X_train_tf, y: Y_train_tf})

            if i % 100 == 0 or i == 0 or i == iterations:
                train_acc = sess.run(accuracy, feed_dict={x: X_train_tf, y: Y_train_tf})
                valid_cost = sess.run(loss, feed_dict={x: X_valid_tf, y: Y_valid_tf})
                valid_acc = sess.run(accuracy, feed_dict={x: X_valid_tf, y: Y_valid_tf})

                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")

        save_path = saver.save(sess, save_path)
        print(f"Model saved in path: {save_path}")

    return save_path
