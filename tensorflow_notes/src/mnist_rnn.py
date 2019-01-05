import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/data/')


def model(n_steps=28, n_inputs=28, n_neurons=150, n_outputs=10,
          n_epochs=100, batch_size=128, beta=0.001,
          learning_rate=0.001, log_dir=None):

    tf.reset_default_graph()

    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], 'X')
    y = tf.placeholder(tf.int32, [None])

    with tf.name_scope('basic_cell'):
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    with tf.name_scope('loss'):
        logits = tf.layers.dense(
            states, n_outputs, name='logits',
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

    regularization_params = np.array([0.001, 0.01, 0.1])
    for b in regularization_params:
        log_dir = '../tf_logs/my_logs_' + str(b)
        model(beta=b, log_dir=log_dir)
