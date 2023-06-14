#!/usr/bin/env python3
"""training proccess"""
import tensorflow as tf
import numpy as np
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
"""train func"""
    tf.reset_default_graph()
    
    # Create placeholders for input and output
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    
    # Build the forward propagation graph
    y_pred = forward_prop(x, layer_sizes, activations)
    
    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    
    # Create the training operation
    train_op = create_train_op(loss, alpha)
    
    # Initialize variables and create a saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    # Start a TensorFlow session
    with tf.Session() as sess:
        sess.run(init)
        
        # Training loop
        for i in range(iterations+1):
            # Run a training step
            _, cost = sess.run([train_op, loss], feed_dict={x: X_train, y: Y_train})
            
            # Print progress after every 100 iterations, 0th iteration, and final iteration
            if i % 100 == 0 or i == 0 or i == iterations:
                train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
                valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
                valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
                
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")
        
        # Save the model
        save_path = saver.save(sess, save_path)
        print(f"Model saved in path: {save_path}")
    
    return save_path
