from datetime import datetime
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


BATCH_SIZE = 100
NUM_EPOCHS = 20
LEARNING_RATE = 0.01


def gen_batch(batch_size):
    """Generate batches of size batch_size
    Arguments:
    batch_size: integer. The number of samples in a minibatch.
    """
    n_batches = int(np.ceil(X_sc.shape[0] / batch_size))
    start = 0
    for batch in range(n_batches):
        end = start + batch_size
        X_batch = X_sc[slice(start, end)]
        y_batch = y_[slice(start, end)]
        yield X_batch, y_batch
        start += batch_size


# Load the California housing data set and normalize it
housing = fetch_california_housing()
X_unsc = housing.data  # Unscaled data
y_ = housing.target.reshape(-1, 1)  # 2D array instead of 1D
ssc = StandardScaler()
X_sc = ssc.fit_transform(X_unsc)
X_sc = np.c_[np.ones(X_sc.shape[0]), X_sc]  # Add the bias column
m, n = X_sc.shape

# Create the log directory for TensorBoard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Define the computational graph
X = tf.placeholder(shape=(None, n), dtype=tf.float32, name='X')
y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform(
    shape=[n, 1], minval=-1, maxval=1), name='theta')
y_pred = tf.matmul(X, theta, name='y_pred')

with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
training_op = optimizer.minimize(mse)

# Initialize the variables (only theta here)
init = tf.global_variables_initializer()

# Save the learned parameters
# saver = tf.train.Saver()

# Save some statistics for TensorBoard
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(NUM_EPOCHS):
        print("epoch = {}".format(epoch))
        batchgen = gen_batch(BATCH_SIZE)
        batch = 0
        step = 0
        for xb, yb in batchgen:
            if batch % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: xb, y: yb})
                file_writer.add_summary(summary_str, step)
                step += 1
            sess.run(training_op, feed_dict={X: xb, y: yb})
            batch += 1
    best_theta = theta.eval()
    print("Best theta: {}".format(best_theta))
    # save_path = saver.save(
    #     sess, '../checkpoints/batch_housing/batch_housing.ckpt')

file_writer.close()
