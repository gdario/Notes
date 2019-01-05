# Network manipulation 

## Saving a model (architecture and weigths)

You can use `keras.model.save(filepath)`. This saves:
- architecture of the model.
- weights.
- training configuration (loss, optimizer).
- state of the optimizer.

To load the model one can use `keras.models.load_model(filepath)`. Given a model stored in `model` we can save it into an HDF5 file as simply as `model.save('my_model.hf')`.

To save only the architecture of a model we can either use `model.to_json()` or `model.to_yaml()`. Note that both commands return a string, which must be assigned to a variable and saved to a file. One can reload the model architecture from the string with `model_from_json(json_string)` or `model_from_yaml(yaml_string)`.

If you only want to save the weights of a model, you can use `model.save_weights('my_model_weights.h5')` and you can load them using `model.load_weights('my_model_weights.h5')`.

Sometimes you need to use only some of the saved weights. For example, we may have saved some weights and we may need to use a subset of them on a different architecture. In this case we need to load the weitghts with the `by_name` option set to `True`. Let's consider a step-by-step case.

1. I create a model, `model1`, for example using the `Sequential` mode. The model contains a couple of hidden layers that I call `hidden_1` and `hidden_2`. Note that you can use the `name='hidden_1'` option both in the sequential and the functional APIs.
2. We train the model and save it with `save_weigths`.
   3. Now I create a second model, `model2`, that contains only one hidden layer with the same number of nodes as `hidden_1` in `model1`. I want this layer to have the the same weights as `hidden_1`. For this to work I need to use the same name the first and only hidden layer of `model_2` that I used for the first hidden layer of `model1`: `hidden_1`.
4. After creating `model2`, I reload the weights from `model_1` using `model.load_weigths` with the option `by_name=True`. Only the weights of `hidden_1` in `model2` will be updated.

## Obtaining the output of an intermediate layer

Let's say we have created a model called `model`. The model contains three hidden layers. We want to extract the output of the second hidden layer, `hidden_2`. The recommended way is to create a new model with the `Model` API. This is the step by step approach.

1. Create the model `model` containing the three hidden layers. Give names to the layers, or at least to the layer you want to collect the output of.
2. Create a new model using `Model`. The input will be `model.input` and the output will be `model.get_layer(layer_name).output`. You can bind this model to a name like `intermediate_layer_model`.
3. To collect the output of the intermediate layer you need to apply the `predict` method to `intermediate_layer_model` by writing something like `intermediate_output = intermediate_layer_model.predict(data)`.

In the [FAQ of the Keras documentation](https://keras.io/getting-started/faq/) you can find another approach based on backends. Below we show a full example:

```py
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

Z_train = X_train.reshape((60000, 784))
Z_test = X_test.reshape((10000, 784))

labels_train = to_categorical(y_train)
labels_test = to_categorical(y_test)

# Create the first model
inputs = Input(shape=(784,), name='input')
hidden1 = Dense(512, activation='relu', name='hidden1')(inputs)
hidden2 = Dense(128, name='hidden2', activation='relu')(hidden1)
outputs = Dense(10, activation='softmax', name='outputs')(hidden2)
model1 = Model(input=inputs, output=outputs)
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_acc', patience=10)
model1.fit(Z_train, labels_train, batch_size=256, nb_epoch=100, validation_data=(Z_test, labels_test), callbacks=[early_stopping])

# Create the intermediate-layer (in this case `hidden2`) model.
model2 = Model(input=model1.input, output=model1.get_layer('hidden2').output)

intermediate_layer_output = model2.predict(Z_test)
print(intermediate_layer_output.shape)
```

## Early Stopping

Introduction early stopping in a model training is very easy:

1. Create the model you want to train.
2. import `EarlyStopping` from the Keras `callbacks` using `from keras.callbacks import EarlyStopping`.
3. Create an `EarlyStopping` instance specifying which quantity we want to monitor and how much patience we have: `early_stopping = EarlyStopping(monitor='val_loss', patience=10)`.
   4. Train the model specifying a fairly large number of epochs but including a `callbacks` option: `model.fit(X, y, ..., callbacks=[early_stopping])`.

**Remember** you still need to specify the number of epochs since the `fit` method has a default value `nb_epoch=10`.

**Important**: when you train a model, the learned weights become part of the model. If you retrain the model, it will restart from where it was left at the end of the previous training. This makes it easy to conditionally stop a training session and modify some parameters, e.g. the learning rate, if certain conditions are met (e.g. the loss is not decreasing anymore).

## Freezing layers

By default layers in a model are trainable. If you want to exclude one layer from training, include the `trainable=False` option. In this case we define the layer and we specify that it is not trainable in the same step. We can also set a layer `trainable` property after it has been instantiated. For example:

```py
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)
```

## Removing layers from sequential models

A sequential model is a list of layers. One can therefore remove a layer using the `pop` method.
