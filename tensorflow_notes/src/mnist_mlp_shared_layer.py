# We analyze the MNIST dataset by first splitting each input into two smaller
# inputs, each containing half of the object. We then create two inputs with
# a shared first hidden layer, and build on top the same architecture used for
# mnist_mlp.py. The goal of this exercise is: 1) understanding how to use
# shared variables and 2) how to visualize weights in TensorBoard
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime


tf.reset_default_graph()

now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
logdir = 'tf_logs/' + now + 'mnist_shared'

n_inputs = 392
n_hidden1 = 150
n_output = 10

learning_rate = 0.005
n_epochs = 50
batch_size = 100

mnist = input_data.read_data_sets('/tmp/data/')


def batch_generator(mnist_input, batch_size):
    n_batches = int(np.ceil(mnist_input.num_examples / batch_size))
    for batch in range(n_batches):
        X_batch, y_batch = mnist_input.next_batch(batch_size)
        X1_batch, X2_batch = np.hsplit(X_batch, 2)
        yield X1_batch, X2_batch, y_batch


he_init = tf.contrib.layers.variance_scaling_initializer()

X1 = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X1')
X2 = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X2')
y = tf.placeholder(tf.int64, shape=None, name='y')

with tf.name_scope('first_layer'):

    hidden1 = tf.layers.dense(X1, n_hidden1, kernel_initializer=he_init,
                              name='hidden1', activation=tf.nn.relu,
                              reuse=None)

    # Extract the weights from the hidden layer
    weights_hidden1 = tf.get_default_graph().get_tensor_by_name(
        'hidden1/kernel:0')
    tf.summary.histogram('weights_hidden1', weights_hidden1)
    hidden2 = tf.layers.dense(X2, n_hidden1, kernel_initializer=he_init,
                              name='hidden1', activation=tf.nn.relu,
                              reuse=True)

with tf.name_scope('second_layer'):
    hidden3 = tf.reduce_mean([hidden1, hidden2], axis=0, name='hidden3')
    logits = tf.layers.dense(hidden3, 10, activation=None, name='output')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(k=1, predictions=logits, targets=y)
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

init = tf.global_variables_initializer()

accuracy_summary = tf.summary.scalar('Training Accuracy', accuracy)
summaries = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(logdir + '_train', tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + '_test', tf.get_default_graph())


with tf.Session() as sess:
    sess.run(init)

    step = 0

    X1_test, X2_test = np.hsplit(mnist.test.images, 2)
    y_test = mnist.test.labels

    for epoch in range(n_epochs):
        print('Epoch: ', epoch)
        batchgen_train = batch_generator(mnist.train, batch_size)
        batchgen_test = batch_generator(mnist.test, batch_size)
        for X1_batch, X2_batch, y_batch in batchgen_train:
            sess.run(training_op,
                     feed_dict={X1: X1_batch, X2: X2_batch, y: y_batch})

            if step % 100 == 0:
                train_summ = sess.run(summaries,
                                      feed_dict={X1: X1_batch,
                                                 X2: X2_batch,
                                                 y: y_batch})
                test_summ = sess.run(summaries,
                                     feed_dict={X1: X1_test,
                                                X2: X2_test,
                                                y: y_test})
                train_writer.add_summary(train_summ, step)
                test_writer.add_summary(test_summ, step)
            step += 1

    train_writer.close()
    test_writer.close()
