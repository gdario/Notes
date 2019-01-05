# We encapsulate a 1D CNN in a function with one argument: the dropout rate.
# The model is run for several values of the dropout rate and a tensorboard
# folder is created with the accuracy on the training and test sets for
# each value.
from datetime import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/')

height = 28
width = 28
channels = 1
n_inputs = height * width

batch_size = 100
n_epochs = 30

# First convolutional layer
conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 2
conv1_pad = "SAME"

# Second convolutional layer
conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

pool1_fmaps = conv2_fmaps

n_fc1 = 64  # N. of units in the fully connected layer
n_outputs = 10


def model(d):
    tf.reset_default_graph()

    # We create the placeholders for the dataset and two convolutional layers
    # with the parameters specified above.
    X = tf.placeholder(shape=(None, n_inputs), dtype=tf.float32, name='X')
    y = tf.placeholder(shape=(None), dtype=tf.int32, name='y')
    training = tf.placeholder(dtype=tf.bool, shape=(), name='dropout')

    # We still need to have a channel dimension
    X_reshape = tf.reshape(X, shape=(-1, n_inputs, 1), name='X_reshape')

    with tf.name_scope('convolutions'):
        conv1 = tf.layers.conv1d(X_reshape, filters=conv1_fmaps,
                                 kernel_size=conv1_ksize, strides=conv1_stride,
                                 padding=conv1_pad)
        conv2 = tf.layers.conv1d(conv1, filters=conv2_fmaps,
                                 kernel_size=conv2_ksize, strides=conv2_stride,
                                 padding=conv2_pad)

    with tf.name_scope('max_pool'):
        pool1 = tf.layers.max_pooling1d(conv2, pool_size=4, strides=4,
                                        padding='same')

    with tf.name_scope('dense'):
        pool1_flat = tf.reshape(pool1, shape=(-1, 98 * 64), name='pool1_flat')
        fc1 = tf.layers.dense(inputs=pool1_flat, units=n_fc1,
                              activation=tf.nn.relu, name="fc1")
        dropout = tf.layers.dropout(fc1, rate=d, name='dropout',
                                    training=training)
        logits = tf.layers.dense(dropout, units=10, name='logits')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    with tf.name_scope('accuracy'):
        correct = tf.nn.in_top_k(k=1, predictions=logits, targets=y)
        accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

    init = tf.global_variables_initializer()

    # Create info for tensorboard
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    train_writer = tf.summary.FileWriter(log_dir + '_train',
                                         tf.get_default_graph())
    test_writer = tf.summary.FileWriter(log_dir + 'test',
                                        tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)
        step = 0
        for epoch in range(n_epochs):
            print('Epoch: {}'.format(epoch))
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                if iteration % 100 == 0:
                    train_acc = accuracy_summary.eval(
                        feed_dict={X: X_batch, y: y_batch, training: True})
                    test_acc = accuracy_summary.eval(
                        feed_dict={X: mnist.test.images, y: mnist.test.labels,
                                   training: False})
                    train_writer.add_summary(train_acc, step)
                    test_writer.add_summary(test_acc, step)
                    step += 1
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch,
                                                 training: True})


for d in [0.3, 0.4, 0.5]:
    # TensorBoard results
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    log_dir = 'tf_logs/mnist' + now + 'd_' + str(d)
    model(d)
