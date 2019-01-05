import tensorflow as tf
import numpy as np


n_inputs = 3
n_neurons = 5
n_steps = 2


seq_length = tf.placeholder(tf.int32, [None])
seq_length_batch = np.array([2, 1, 2, 2])

X_batch = np.array([
        [[0, 1, 2], [9, 8, 7]],  # instance 0
        [[3, 4, 5], [0, 0, 0]],  # instance 1
        [[6, 7, 8], [6, 5, 4]],  # instance 2
        [[9, 0, 1], [3, 2, 1]],  # instance 3
    ])

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], 'X')
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(
    basic_cell, X, dtype=tf.float32, sequence_length=seq_length
)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    outputs_val, states_val = sess.run(
        [outputs, states],
        feed_dict={X: X_batch, seq_length: seq_length_batch}
    )

    print(outputs_val)
