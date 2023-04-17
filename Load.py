import keras
import numpy as np
import pandas as pd
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from numpy import nan
import nltk
import sklearn.feature_extraction.text as pre
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pk



data=pd.read_csv("spamPretraité.csv")
Y = data["label"].to_numpy()
Y[Y == "ham"]=0
Y[Y == "spam"]=1
#X = preprocessing.normalize(X)
#print(X)
X=pd.read_csv("countMatrix")
print(X)
X= X.to_numpy()
# Convert label data to binary values
# y = pd.get_dummies(data['Spam']).values
X_train = X[0:3300]
X_test = X[3300:4197]
Y_train = Y[0:3300]
Y_test = Y[3300:4197]
from sklearn.preprocessing import MinMaxScaler
# Normalize data


print(X_train)

print(Y_train.shape)
print(Y_test.shape)





from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

X_train = np.asarray(X_train).astype('float32')
X_train = tensorflow.expand_dims(X_train, axis=-1)
X_train = tensorflow.expand_dims(X_train, axis=-1)


Y_train = np.asarray(Y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
X_test = tensorflow.expand_dims(X_test, axis=-1)
X_test = tensorflow.expand_dims(X_test, axis=-1)
Y_test = np.asarray(Y_test).astype('float32')
print(X_train.shape)
print(X_test.shape)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=X_train.shape[1:],padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(16, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))


model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))


model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))


model.add(Flatten())  # Flattening the 2D arrays for fully connected layers

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(364, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_test, Y_test),batch_size=32)