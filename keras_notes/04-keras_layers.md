# Keras Layers

## Common methods

There are many different types of layer, but some methods are common to all of them. We have already encountered these methods but for whole models rather than individual layers.

- `layer.get_weigths()`: returns the weigths of a layer as a list of Numpy arrays.
- `layer.set_weights(weights)`: sets the weights of the layer from a list of Numpy arrays.
- `layer.get_config()`: returns a dictionary with the configuration of the layer.

When we mentioned the concept of layer nodes we noticed that the input (shape) and output (shape) must be extracted with different methods depending on the layer having a single node or multiple ones.

| One node | Multiple Nodes |
|----------|----------------|
| `layer.input` | `layer.get_input_at(node_index)` |
| `layer.output` | `layer.get_output_at(node_index)` |
| `layer.input_shape` | `layer.get_input_shape_at(node_index)` |
| `layer.output_shape` | `layer.get_output_shape_at(node_index)` |

## Dense Layers

``` python
keras.layers.core.Dense(output_dim, init='glorot_uniform', activation=None,
	weights=None, W_regularizer=None, b_regularizer=None,
	activity_regularizer=None, W_constraint=None, b_constraint=None,
	bias=True, input_dim=None)
```

Most parameters have self-explanatory names. `init` specifies the type of initialization function for the weights. There are many possible different initializations (`uniform`, `lecun_uniform`, `normal`, `identity`, `orthogonal`, `zero`, `one`, `glorot_normal`, `glorot_uniform`, `he_normal`, `he_uniform`). You can find a discussion on various initialization methods in [this blog post](http://deepdish.io/2015/02/24/network-initialization/).

If you pass a `weight` argument in the form of a list of Numpy array, these will be the initial weights for the model. The list must have two elements: one, for the weights, of shape `(input_dim, output_dim)` and one, for the biases, of shape `(output_shape,)`

The `W_regularizer` argument specifies the type of `WeightRegularizer` to be applied to the weights and/or the biases. These are instances of L1 or L2 regularizations. The penalties are applied on a per-layer basis. If you want to apply a `W_regularizer` in a model you can write something like:

``` python
from keras.regularizers import l2, activity_l2
model.add(Dense(64, input_dim=64, W_regularizer=l2(0.01), 
	activity_regularizer=activity_l2(0.01)))
```

In the example above we have applied an L2 regularizer to the weights, but also one to the activities. The `activity_regularizer` argument specify the type of `ActivityRegularizer` instance we want to use. The available methods are `l1`, `l2` and `l1l2` (`activty_l1`, `activty_l2` and `activty_l1l2` for the activities). The `l1l2` methods require two parameters.

`W_constraint` and `b_constaint` allow imposing certain constraints to the weights and/or the bias. There are three available constraints:

- `maxnorm`: maximum-norm constraint. Example `W_constraint=maxnorm(2)`.
- `nonneg()`: non-negativity constraint.
- `unitnorm()`: unit-norm constraint. Enforces the matrix to have unit norm along the last axis.

The `maxnorm` constraint is a recommendation from the [original Dropout paper](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) (bottom of page 1934), where a constraint is enforced on the weights of each hidden layer, making sure that the maximum norm of the weights does not exceed 3. Here the default is 2.

## Activation Layers

`keras.layers.core.Activation(activation)`

Here `activation` is one of the available activation functions. These are:

- `softplus`.
- `softsign`.
- `relu`.
- `tanh`.
- `sigmoid`.
- `hard_sigmoid`.
- `linear`.

In addition there are some advanced activations in the module `keras.layers.advanced_activations`. These comprise:

- `LeakyRelu`: Leaky ReLu. $f(x) = \alpha x$ for $x < 0$. $f(x) = x$ for $x \ge 0$.
- `PReLU`: (parametric rectified ReLU: $f(x) = \alpha x$ for $x < 0$. $f(x) = x$ for $x \ge 0$.
- `ELU`: Exponential Linear Unit $f(x) = \alpha (\exp(x) - 1)$ for $x < 0$. $f(x) = x$ for $x \ge 0$.
- `ParametricSoftplus`: $f(x) = \alpha \log(1 + \exp(\beta x))$.
- `ThresholdedReLU`: $f(x) = x$ for $x > \theta$. $f(x) = 0$ otherwise.
- `SReLU`: S-shaped ReLU. No functional form available in the documentation of Keras.

## Dropout Layer

The dropout layer has only one parameter `p`, ther fraction of the input (relative to the current layer) units to drop.

## Flatten Layer

Flattens the input. It does not take arguments. Useful for example when transitioning from a convolutional layer to a fully connected one.

``` python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
            border_mode='same',
            input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)
model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

## Reshape Layer

It reshapes a layer to a shape specified in the `target_shape` tuple. In a Sequential model it can be added as any other layer, but its effect is to reshape the output of the previous layer.

``` python
# as first layer in a Sequential model
model = Sequential()
# Reshape is applied directly to the input shape
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)
```

## Permute Layer

Given an input layer, it permutes its dimensions in the order specified in the `dims` argument. Apparently it's useful when connecting CNNs and RNNs together. Note that the indexing in `dims` starts at 1, not at 0. For example `(2, 1)` exchanges the first and the second dimension of the output.


## Repeat Layer

Repeat the input n times. It takes a 2D tensor as an input with shape `(nb_samples, features)` and returns a 3D tensor of shape `(nb_samples, n, features)`.

## Merge Layer

```py
keras.engine.topology.Merge(layers=None, mode='sum', concat_axis=-1,
	dot_axes=-1, output_shape=None, output_mask=None, arguments=None,
	node_indices=None, tensor_indices=None, name=None)
```

Merges a list of tensors into one tensor. The merging mode is specified by the `mode` argument. This can be `sum`, `mul`, `concat`, `ave`, `cos`, `dot`, `max` or a lambda. If using `dot` or `cos` one must specify the `dot_axes`, i.e. the integer or tuple of axes to use in the operation.

If one or more input layers have multiple nodes, we must specify the `node_indices`. If some input layer node returns multiple tensors, we must specify `tensor_indices`.

## Lambda Layer

A `Lambda` layer allows the evaluation of an arbitrary function on an input tensor. For example, if we want to add a layer that takes the square of each component of the previous layer, we can type `model.add(Lambda(lambda x: x ** 2))`. The function is specified in a `lambda` expression.

**TODO** understand the `antirectifier` example.

## ActivityRegulation Layer

This layer does not modify its input, but it updates the cost function based on the activity of the input. It has the following form:

`keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)`

## Masking Layer

This layer is used in connection with LSTM layers, and is used to skip selected time-steps of the input sequence. For example, we have a collection `x` of samples where a certain number of features have been measured in a certain number of time-steps. Our input tensor has shape `(samples, timesteps, features)`. We may be missing values for two of the time-steps, say 3 and 5. We can fill the values for these time-steps with zeros, and use a `Masking` layer to skip those steps.

``` python
x[:, 3, :] = 0
x[:, 5, :] = 0
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(timesteps, features)))
model.add(LSTM(32))
```

``` python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

## Highway Layer 

```py
keras.layers.core.Highway(init='glorot_uniform', activation=None,
	weights=None, W_regularizer=None, b_regularizer=None,
	activity_regularizer=None, W_constraint=None, b_constraint=None, 
	bias=True, input_dim=None)
```
[Highway Networks](https://arxiv.org/pdf/1505.00387v2.pdf) are densely connected networks. 

**TODO** Understand how highway networks work complete this part.

## MaxoutDense Layer 

**TODO** Understand how Maxout networks work complete this part.
