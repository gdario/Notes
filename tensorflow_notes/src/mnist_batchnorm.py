import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10

learning_rate = 0.01
n_epochs = 120
batch_size = 100

mnist = input_data.read_data_sets('/tmp/data/')

X = tf.placeholder(tf.float32, (None, n_inputs), 'X')
y = tf.placeholder(tf.int64, (None), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')

he_init = tf.contrib.layers.variance_scaling_initializer()


# Define the SELU non-linearity
def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))


with tf.name_scope('model'):
    hidden1 = tf.layers.dense(X, n_hidden1, kernel_initializer=he_init,
                              name='hidden1')
    bn1 = tf.layers.batch_normalization(hidden1, training=training,
                                        momentum=0.9)
    bn1_act = tf.nn.elu(bn1)

    hidden2 = tf.layers.dense(bn1_act, n_hidden2, kernel_initializer=he_init,
                              name='hidden2')
    bn2 = tf.layers.batch_normalization(hidden2, training=training,
                                        momentum=0.9)
    bn2_act = tf.nn.elu(bn2)
    logits_before_bn = tf.layers.dense(bn2_act, 10, name='output')
    logits = tf.layers.batch_normalization(logits_before_bn,
                                           training=training,
                                           momentum=0.9)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1, 'correct')
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:

    sess.run(init)
    n_batches = int(np.ceil(mnist.train.num_examples / batch_size))

    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops],
                     feed_dict={X: X_batch, y: y_batch})

        train_acc = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                 y: mnist.test.labels})
        print("Epoch:", epoch, "Train acc.:", train_acc,
              "Test acc.:", test_acc)
