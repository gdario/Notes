# About Keras Models

## Methods common to the sequential and the functional APIs

Some general methods are in common:

- `model.summary()`: returns a textual description of the model.
- `mode.get_config()`: returns a dictionary containing the configuration of a model. Let's see a simple example below:

```py
inputs = Input(shape=(784,))
outputs = Dense(10, activation='softmax')(inputs)
model = Model(input=inputs, output=outputs)
cfg = model.get_config()
cfg.keys()
```

The output of the code snippet above is something like this:

```
dict_keys(['name', 'layers', 'input_layers', 'output_layers'])
```

If we inspect `cfg['layers']` we obtain a list of dictionaries like the one shown below:

```
In [21]: cfg['layers']
Out[21]:
[{'class_name': 'InputLayer',
  'config': {'batch_input_shape': (None, 784),
   'input_dtype': 'float32',
   'name': 'input',
   'sparse': False},
  'inbound_nodes': [],
  'name': 'input'},
 {'class_name': 'Dense',
  'config': {'W_constraint': None,
   'W_regularizer': None,
   'activation': 'softmax',
   'activity_regularizer': None,
   'b_constraint': None,
   'b_regularizer': None,
   'bias': True,
   'init': 'glorot_uniform',
   'input_dim': 784,
   'name': 'dense_1',
   'output_dim': 10,
   'trainable': True},
  'inbound_nodes': [[['input', 0, 0]]],
  'name': 'dense_1'}]
```

Other methods that are common to the sequential and the functional API are:

- `model1.get_weights()`: returns a list of weight tensors as Numpy arrays.
- `model2.set_weights(weights)`: sets the values of the weigths of the model from a list of Numpy arrays (produced by `model1.get_weigths()`). `model2` must have the same shape. We have seen in section 2 that to extract the output of an intermediate layer you need to do something like this:

```py
intermediate_model = Model(input=model1.input,
	output=model1.get_layer('hidden2').output)
intermediate_layer_output = model2.predict(Z_test)
```
In other words, it's not enough to use `get_layer`. You also need to extract the `output` from that layer.

In section 2 we have also seen the `get_weights`, `set_weights`, `save_weights` and `load_weights` methods. These also apply to both APIs.

## The Sequential API

Let `model` be a model created using the sequential API. Each instance has attributes and methods. A sequential model is a list of layers, and there are multiple, equivalent ways of constructing a model. We can create a model in one single step as a list, as shown below.

``` python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

Or we can create it using the `add` method to add new layers (or the `pop` method, if we want to remove layers).

``` python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

Note that the above is equivalent to writing

``` python
model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))
```

### Attributes of the sequential API

`model.layers` returns a list of the layers added to the model.

### Methods of the sequential API

The sequential API has the following methods:

- `compile`.
- `fit`.
- `evaluate`.
- `predict`, `predict_classes`, `predict_proba`.
- `train_on_batch`, `test_on_batch`, `predict_on_batch`.
- `fit_generator`, `evaluate_generator`, `predict_generator`.

Let's look at each of these more into detail.

#### The `compile` method

`compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)`

Here the only argument that is not immediately obvious is `sample_weight_mode`.
These weights are Numpy arrays used to scale the loss function during the training step. These weights can be one-dimensional or two-dimensional. In the 1D case there is a 1-to-1 mapping between weights and samples. Each sample is weighted differently during the training phase. The 2D case is used when dealing with temporal data, and implies the use of different weights at each time-step on each sample. To decide which of these two mechanisms we want to use, we must set the `sample_weight_mode`. There are two values for this argument: `temporal` and `None`.

### The `fit` method

The `fit` method has a number of arguments, most of which self-explanatory.

```py
fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
	validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
	sample_weight=None, initial_epoch=0)
```

The `callbacks` argument is a *list* of `keras.callbacks.Callback` instances. Some of these are automatically applied to every model (logging, history), others can be called by the user. For example `ModelCheckpoint` allows to automatically save the best model so far. We have already seen `EarlyStopping`. There is a `LearningRateScheduler` that takes an epoch index as input and returns a new learning rate as output. Similarly there is a `ReduceLROnPlateau` callback that reduces the learning rate when a metric has stopped improving.
There's also a `TensorBoard` callback that writes a log for TensorBoard.

We have seen earlier on the meaning of the `sample_weight` parameter. There is also a `class_weight` parameter. This is a *dictionary* mapping classes to weight values. One interesting comment can be found in this [post on the Keras user mailing list](https://groups.google.com/forum/#!topic/keras-users/MUO6v3kRHUw). A user asked what are the best weights to use when you have unbalanced classes. Apparently there is a function `scikit-learn.utils.compute_class_weight` that does exactly that. Note, however, that the function is not documented. You can see it [in the GitHub repo](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/class_weight.py#L12).

#### The `evaluate` method

The `evaluate` method computes the loss function batch by batch.

`evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)`

If the model has no metrics the returned value will be the scalar loss on the test set. If the model has additional metrics, the function will return a list of scalars. There is an additional attribute `model.metrics_names` that show the labels for the scalar outputs.

#### `predict`, `predict_classes`, `predict_proba` methods

Each of these methods operates batch by batch.

#### `train_on_batch`, `test_on_batch`, `predict_on_batch` methods

Both these methods accept a `sample_weight` argument. `train_on_batch` also accepts the `class_weight` argument. `predict_on_batch` returns the predictions for a single batch of samples.

#### The `fit_generator` method

This method has a few arguments in common with the `fit` method. It fits the model on data generated by a generator, operating batch by batch. One advantage is that one can do real-time data augmentation on images on the CPU while the model is training on the GPU.

``` python
fit_generator(self, generator, samples_per_epoch, nb_epoch, 
	verbose=1, callbacks=None, validation_data=None, nb_val_samples=None,
	class_weight=None, max_q_size=10, nb_worker=1, pickle_safe=False,
	initial_epoch=0)
```

The `generator` argument must return either a tuple `(inputs, targets)` or `(inputs, targets, sample_weights)`. Note that all arrays should contain the same number of samples. The generator will loop over its data indefinitely.
There is a `max_q_size` parameter defining the maximum size of the generator queue. This is by default set to 10.

The structure of the `validation_data` tuple is the same as `generator`, i.e., it can be a tuple of `(inputs, targets)` and optionally include `sample_weights`. It can also be a generator. In this case, the `nb_val_samples` parameter specifies how many validation samples define an epoch for the validation set.

An epoch finishes when `samples_per_epoch` samples have been seen by the model.

##### Creating a generator from a text file

The example below assumes that we have a function called `process_line` that processes each line of a text file returning the values and the targets.

``` python
def generate_arrays_from_file(path):
    while 1:
    f = open(path)
    for line in f:
        # create Numpy arrays of input data
        # and labels, from each line in the file
        x, y = process_line(line)
        yield (x, y)
    f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, nb_epoch=10)
```

One can thing extensions that return batches of lines, or that operate on HDF5 files.

An even more instructive example comes from the CIFAR10 example. Here we imagine to have an ImageNet() generator that produces batches of 10K images. These are loaded into memory by the `for` loop and processed by `model.fit` in batches of 32 images.

``` python
for e in range(nb_epoch):
    print("epoch %d" % e)
    for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
        model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
```

#### `evaluate_generator`, `predict_generator` methods

These two functions have the same argument. `predict_generator` generates prediction for the samples produced by a data generator. The generator should return the same kind of data accepted by `predict_on_batch`.

## The Model class API ##

### Attributes of the Model API ###

The `Model` API combines together all the layers required for a computation. For example, if we have the following layers, `inputs`, `hidden_1`, `hidden_2`, `outputs`, the final model instance will be created as `model = Model(input=inputs, output=outputs)`. The model will take care of concatenating the other pieces, as they are referred to by the downstream layers.

If there are multiple inputs or multiple outputs, we must use lists. The attributes of the `Model` API are:

- `model.layers`: a flattened list of the layers in the graph.
- `model.inputs`: a list of input tensors.
- `model.outputs`: a list of output tensors.

### Methods of the Model API ###

The methods are the same as for the Sequential API. There is an additional method, `get_layer` that returns a layer bsaed on its name or its index in the graph.

`get_layer(self, name=None, index=None)`
