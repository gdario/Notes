# import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/data/')


def model(n_steps=28, n_inputs=28, n_layers=3, n_neurons=100,
          n_outputs=10, n_epochs=100, batch_size=128,
          beta=0.001, learning_rate=0.001, log_dir=None):
    """Model template for a multi-layer recurrent network

    Parameters:
    n_steps: integer. The number of time points.
    n_inputs: integer. The dimension of the input sequence.
    n_layers: integer. The number of recurrent layers.
    n_neurons: integer. The number of hidden neurons in a cell.
    n_outputs: integer. The number of output neurons.
    n_epochs: integer. The number of epochs in the training phase.
    batch_size: integer. The mini-batch size.
    beta: real. The beta regularization parameter.
    learning_rate: real. The learning rate for the Adam optimizer.
    log_dir: string. The directory where TensorBoard will store its logs.
    """

    tf.reset_default_graph()

    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], 'X')
    y = tf.placeholder(tf.int32, [None])

    with tf.name_scope('multi_layer_cell'):
        layers = [tf.contrib.rnn.BasicRNNCell(n_neurons, tf.nn.relu)
                  for layer in range(n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
        outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X,
                                            dtype=tf.float32)
        states_concat = tf.concat(states, axis=1)

    with tf.name_scope('loss'):
        logits = tf.layers.dense(
            states_concat, n_outputs, name='logits',
            kernel_initializer=tf.variance_scaling_initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(beta)
        )
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits, name='xentropy'
        )
        base_loss = tf.reduce_mean(xentropy, name='base_loss')
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([base_loss] + reg_losses, name='loss')

    with tf.name_scope('training'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('accuracy'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    train_writer = tf.summary.FileWriter(log_dir + '_train',
                                         tf.get_default_graph())
    test_writer = tf.summary.FileWriter(log_dir + 'test',
                                        tf.get_default_graph())

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

                if iteration % 100 == 0:
                    train_acc = accuracy_summary.eval(
                        feed_dict={X: X_batch, y: y_batch}
                    )
                    train_writer.add_summary(train_acc, iteration)
                    test_acc = accuracy_summary.eval(
                        feed_dict={X: X_test, y: y_test}
                    )
                    test_writer.add_summary(test_acc, iteration)


if __name__ == '__main__':
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    log_dir = '../tf_logs/my_logs_' + now
    model(log_dir=log_dir)
