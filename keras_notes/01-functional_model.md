# The Keras Functional API

The main characteristic of the Functional API is that all models are callable, just like layers. Each layer can become the input of another layer, and a whole method itself \(via its output\) can become the input of another model. This becomes particularly important when doing \_transfer learning. \_In the simplest case you start stacking layers passing the previous one as an argument to the next one.

```py
from keras.layers import Input, Dense
from keras.models import Model

# this returns a tensor
inputs = Input(shape=(784,)) # input layer

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x) # output layer

model = Model(input=inputs, output=predictions)
```

Here we have used the `Input` and `Dense` layers. The activations are specified inside the layer call. In other words, `x` is an instance of the `Dense` class with  he output shape and the type of activation specified during the instantiation. 

It is possible to treat a trained model as if it were a layer by calling it on a tensor. When we do so, we are using the architecture _and_ the learned weights. If we compile and fit `model` above, we can then create a new input layer and apply the model to it, as shown below.

```py
x = Input(shape=(784,))
y = model(x)
```

Whenever we call a layer on an input, we create a new tensor: the output of the layer. In the example above `y` contains the output of `model`. There is another important effect: calling a layer on some input adds a node to that layer linking the input tensor to the output tensor.

The result of this operation is that `y` will return the softmax output of `model`.

## Multi-input and multi-output models

There are several things that I don't understand.

1. Embeddings.
2. The regularizing effect of the auxiliary output.

## Shared Layers

In the example in the Keras documentation we want to predict whether two tweets are from the same person. We build a model that encodes the tweets into two vectors, concatenates them and adds a logistic regression on top. The output of the logistic regression is the probability that the two inputs come from the same person.

The idea is that if we use an LSTM to encode the first tweet, we can use the same layer for the second tweet. The two inputs have the same structure, i.e. the same input and output size.

```py
from keras.layers improt Input, LSTM, Dense, merge
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
```

We want to share the same LSTM layer between the two inputs.

```py
shared_lstm = LSTM(64)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)
```

Here we have created an LSTM layer that returns a vector of length 64 and takes as input `tweet_a` and `tweet_b`. Once the two inputs have been encoded we can merge them.

```
merged_vector = merge([encoded_a, encoded_b], mode='concat', concat_axis=-1)
```

We said before that calling a layer on an input returns a new tensor containing the output of the layer, and adds a node to the layer linking the input to the output. If we call the same layer multiple times, that layer owns multiple nodes indexed in increasing order.

To extract the output of a layer, you can use the `output` method. Let's consider a case similar to the one above.

```py
a = Input(shape=(140, 256))
b = Input(shape=(140, 256))
lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)
```

Here the `lstm` layer has two nodes connecting it two `encoded_a` and `encoded_b`. To access the correct node we need to index using `get_output_at`

```py
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

Other properties depend on the particular node we are looking at. For example, the input and output shapes. If the layer has only one node, we can use `layer.output_shape` and `layer.input_shape`, otherwise we need `layer.output_shape_at` and `layer.input_shape_at`.

