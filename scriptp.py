# This is a sample Python script.
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



# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    print_hi('PyCharm')
    data = pd.read_csv("email_data_spam.csv")
    print(data.loc[223]["email"])
    print(prepro(data.loc[346]["email"]))
    print(data[data["email"] == data.loc[223]["email"]].size)
    print(data[data["email"] == ""].size)
    for i in range(1396):
        data.loc[i]["email"]=prepro((str)(data.loc[i]["email"]))

    print(data[data["email"] == data.loc[223]["email"]].size)

    print(data[data["email"]==""].size)
    matest = np.array(data["email"])

    matr = vecvec.fit_transform(np.array(data["email"]))
    matr = pd.DataFrame(matr,columns= vecvec.get_feature_names_out())
    print(matr)

    data.to_csv("spamPretraité.csv")

    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    # Créer un modèle séquentiel
    model = Sequential()

    # Ajouter des couches de convolution et de max pooling
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Ajouter une couche de flattening
    model.add(Flatten())

    # Ajouter des couches fully connected
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle sur les données d'entraînement
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))'''
#,index = [i for i in range(3947,4197)]
#,index=[i for i in range(1396,3947)]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
'''data = pd.read_csv("email_data_spam.csv")

data2 = pd.DataFrame(pd.read_csv("preprocessed_emails.csv"))
data2.rename(columns={"preprocessed_email": "email"}, inplace=True)

print(data2)
data3 = pd.DataFrame(pd.read_csv("preprocessed_emails_hard_ham.csv"))
data3.rename(columns={"preprocessed_email": "email"}, inplace=True)
print(data3)
print('here')

for i in range(1396):
    data.loc[i]["email"]=prepro((str)(data.loc[i]["email"]))
data =pd.concat([data3,data,data2],ignore_index=True)
data = data.sample(frac=1).reset_index()
data.to_csv("spamPretraité.csv")'''
data = pd.read_csv("spamPretraité.csv")
data["email"]=data["email"].astype(str)
Y = pd.DataFrame(np.array(data["label"]),columns=["label"])
Y[Y["label"] == "spam"] = 1
Y[Y["label"] == "ham"] = 0
Y_train = np.array((Y.iloc[[i for i in range(0,3300)]]))
Y_test = np.array((Y.iloc[[i for i in range(3300,4197)]]))
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(data["email"].values)
sequences = tokenizer.texts_to_sequences(data["email"])
word_index = tokenizer.word_index
vecvec = CountVectorizer()
datdat =vecvec.fit_transform(data['email'].values).toarray()
print(datdat)
print(datdat.shape)
datata = pd.DataFrame(datdat,index = [i for i in range(datdat.shape[0])],columns=vecvec.get_feature_names_out())
print(datata)
summ = datata.sum(axis=0)
print(summ)
listvocab= summ[summ>3]
datata = datata[listvocab.index]
datata[datata>1]=1
print(datata)
print(listvocab)
file = open('vocabulaire', 'wb')
datata.to_csv("countMatrix")
pk.dump(listvocab,file)

file.close()
'''
X = tokenizer.texts_to_sequences(data['email'].values)
X = pad_sequences(X)
#X = preprocessing.normalize(X)
#print(X)

# Convert label data to binary values
# y = pd.get_dummies(data['Spam']).values
print('Found %s unique tokens.' % len(word_index))
X_train = X[0:3300]
X_test = X[3300:4197]
from sklearn.preprocessing import MinMaxScaler
# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

print(Y_train.shape)
print(Y_test.shape)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

# Créer un modèle séquentiel
model = Sequential()

# Ajouter des couches de convolution et de max pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:],padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))


# Ajouter une couche de flattening
model.add(Flatten())

# Ajouter des couches fully connected
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(X_train.shape)
# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test),batch_size=32)'''
