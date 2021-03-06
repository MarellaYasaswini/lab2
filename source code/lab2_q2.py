import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.callbacks import TensorBoard
df = pd.read_csv('heart.csv')
df.info()
label = df['target']
features = df.drop(['target'],axis=1)
X, X_test, y, y_test = train_test_split(features,label,test_size=0.2,train_size=0.8,random_state=5)
X_train, X_validation, y_train, y_validation = train_test_split(X,y,test_size = 0.25,train_size=0.75,random_state=5)

s = StandardScaler()
X_train_scaled = s.fit_transform(X_train)
X_validation_scaled = s.fit_transform(X_validation)
X_test_scaled  = s.fit_transform(X_test)

# Make one -hot encoder
def one_hot_encode_object_array(arr):
  uniques, ids = np.unique(arr, return_inverse=True)
  return np_utils.to_categorical(ids, len(uniques))

# convert (*,) -> (*,2)
y_train_ohe = one_hot_encode_object_array(y_train)
y_validation_ohe = one_hot_encode_object_array(y_validation)
y_test_ohe = one_hot_encode_object_array(y_test)

from keras.activations import relu, elu, selu, sigmoid, exponential, tanh

#     'activation': [relu, elu, selu, sigmoid, exponential, tanh],
#     'batch_size': [64, 128, 256],
#     'epochs'    : [50, 100, 150, 200]
#     'optimizer' : [adam, sgd]

# Hyper Parameters
optimizer  = 'adam'
activation = 'sigmoid'
epochs = 90
batch_size = 100

model = Sequential()
model.add(Dense(32, input_shape=(15,)))
model.add(Activation(activation))
model.add(Dense(16, input_shape=(11,)))
model.add(Activation(activation))
model.add(Dense(64, input_shape=(12,)))
model.add(Activation(activation))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

tensorboard = TensorBoard(log_dir='1', histogram_freq=0, write_graph=True, write_images=False)
history=model.fit(X_train_scaled, y_train_ohe, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_validation_scaled, y_validation_ohe),callbacks=[tensorboard])

fig1 = plt.figure()
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves : ', fontsize=16)
fig1.savefig('loss_lstm.png')

score, accuracy = model.evaluate(X_test_scaled, y_test_ohe, batch_size=batch_size, verbose=0)
print("Test fraction correct (NN-Score) = {:.2f}".format(score))
print("Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))