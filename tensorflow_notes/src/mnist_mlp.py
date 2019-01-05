import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from datetime import datetime
# from sklearn.metrics import accuracy_score

# Create the log folder for TensorBoard
# now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
logdir = "tf_logs/mnist_tutorial"

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


def hidden_layer(X, n_units, name, activation=None):
    """Create a hidden layer with a give activation
    """
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_units), stddev=stddev)
        # init = tf.random_uniform((n_inputs, n_units), minval=-1, maxval=1,
        #                          dtype=tf.float32)
        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_units]), name='biases')
        z = tf.matmul(X, W) + b
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', z)
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z


with tf.name_scope('dnn'):
    hidden1 = hidden_layer(X, 200, 'hidden1', 'relu')
    hidden2 = hidden_layer(hidden1, 100, 'hidden2', 'relu')
    logits = hidden_layer(hidden2, 10, 'output')

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

train_writer = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
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
    train_writer.close()
    test_writer_close()
