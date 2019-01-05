from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from datetime import datetime
# from sklearn.metrics import accuracy_score

# Create the log folder for TensorBoard
now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
logdir = "tf_logs/selu"

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10

learning_rate = 0.01
n_epochs = 120
batch_size = 50

mnist = input_data.read_data_sets('/tmp/data/')

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


def selu(z, scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))


activ = selu

# import pdb; pdb.set_trace()

# He initialization
he_init = tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(inputs=X, units=n_hidden1, activation=activ,
                              name='hidden1', kernel_initializer=he_init)
    hidden2 = tf.layers.dense(inputs=hidden1, units=n_hidden2,
                              activation=activ, name='hidden2',
                              kernel_initializer=he_init)
    logits = tf.layers.dense(inputs=hidden2, units=10, activation=None,
                             name='output')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

train_writer = tf.summary.FileWriter(logdir + 'train/',
                                     tf.get_default_graph())
test_writer = tf.summary.FileWriter(logdir + 'test/', tf.get_default_graph())

accuracy_summary = tf.summary.scalar('Accuracy', accuracy)
global_summary = tf.summary.merge_all()

# Execution phase
with tf.Session() as sess:

    sess.run(init)
    n_batches = mnist.train.num_examples // batch_size

    for epoch in range(n_epochs):

        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train, train_summ, s = sess.run(
            [accuracy, accuracy_summary, global_summary],
            feed_dict={X: X_batch, y: y_batch})
        train_writer.add_summary(s, epoch)

        acc_test, test_summ = sess.run([accuracy, accuracy_summary],
                                       feed_dict={X: mnist.test.images,
                                                  y: mnist.test.labels})
        test_writer.add_summary(test_summ, epoch)

        print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)

    save_path = saver.save(sess, '../output/mnist_final_model.ckpt')
