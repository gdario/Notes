import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


tf.reset_default_graph()

# TensorBoard results
log_dir = 'tf_logs/mnist_cnn1d'

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
conv1_stride = 1
conv1_pad = "SAME"

# Second convolutional layer
conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool1_fmaps = conv2_fmaps

n_fc1 = 64  # N. of units in the fully connected layer
n_outputs = 10


# We create the placeholders for the dataset and two convolutional layers
# with the parameters specified above.
X = tf.placeholder(shape=(None, n_inputs), dtype=tf.float32, name='X')
y = tf.placeholder(shape=(None), dtype=tf.int32, name='y')

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
    dpout = tf.layers.dropout(fc1, rate=0.4, name='dropout')
    logits = tf.layers.dense(dpout, units=10, name='logits')

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

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train,
              "Test accuracy:", acc_test)
