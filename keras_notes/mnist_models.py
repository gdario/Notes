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

early_stopping = EarlyStopping(monitor='val_acc', patience=10, mode='max')
model1.fit(Z_train, labels_train, batch_size=256, nb_epoch=100, validation_data=(Z_test, labels_test), callbacks=[early_stopping])

# Create the intermediate-layer (in this case `hidden2`) model.
model2 = Model(input=model1.input, output=model1.get_layer('hidden2').output)

intermediate_layer_output = model2.predict(Z_test)
