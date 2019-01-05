from keras.datasets import mnist


def process_dataset(x, y):
    """
    Transform the keras version of mnist into numpy arrays suitable for TF.
    :param x: the numpy array with the pixel intensities.
    :param y: the numpy array with the class labels.
    :return: a tuple with x, y converted and reshaped.
    """
    x = x.astype('float32') / 255.
    y = y.astype('int32').reshape(-1, 1)
    return x, y


def create_train_and_dev_set(x, y, n_dev_samples=10000):
    """
    Split the training set into a training and a dev set.
    :param x: the pixel intensities in the training set (60000, 28, 28)
    :param y: the class labels in the training set (60000, 1)
    :param n_dev_samples:
    :return:
    """
    n_train_samples = x.shape[0]
    new_n_train_samples = n_train_samples - n_dev_samples

    x_dev_set = x[new_n_train_samples:]
    y_dev_set = y[new_n_train_samples:]

    x_new_train_set = x[:new_n_train_samples]
    y_new_train_set = y[:new_n_train_samples]

    return (x_new_train_set, y_new_train_set), (x_dev_set, y_dev_set)


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data('../data/mnist.npz')

# Transform the training and test sets in proper numpy arrays
x_train, y_train = process_dataset(x_train, y_train)
x_test, y_test = process_dataset(x_test, y_test)

# Generate a dev set
(x_train, y_train), (x_dev, y_dev) = create_train_and_dev_set(x_train, y_train)
