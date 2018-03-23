import os
import re
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense , LSTM
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras import optimizers
from numpy import zeros
from numpy import asarray 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers.core import Flatten
import keras.utils
import nltk
import random

os.getcwd()

os.chdir('C:\\Users\\Hanu\\Documents\\insofe\\keggal\\toxic comments')

toxic = pd.read_csv("train.csv", header =0)
toxic_test = pd.read_csv("test.csv", header =0)
print (toxic)

print (toxic.head(5))
print (toxic.tail(5))

toxic_comments = toxic["comment_text"]

toxic.dtypes
toxic_test.dtypes


toxic_comments.dtypes


for col in ['comment_text']:toxic[col] = toxic[col].astype('category')
for col in ['comment_text']:toxic_test[col] = toxic_test[col].astype('category')
toxic.dtypes

class_var = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
y = toxic[class_var].values

toxic_comments_train = toxic['comment_text']
toxic_comments_test = toxic_test['comment_text']


#stopwords

from nltk.corpus import stopwords
set(stopwords.words('english'))
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
words_tokened = word_tokenize(str(toxic_comments))

filtered_sentence = [w for w in words_tokened if not w in stop_words]
 
filtered_sentence = []
 
for w in words_tokened:
 if w not in stop_words:
        filtered_sentence.append(w)
 
print(words_tokened)
print(filtered_sentence)

#stemming

toxic_comment_train = toxic["comment_text"].astype(str)
toxic_comment_train = re.sub('[^a-zA-Z]', ' ', str(toxic_comment_train))
toxic_comment_train = toxic_comment_train.lower()

toxic_comment_test = toxic["comment_text"].astype(str)
toxic_comment_test = re.sub('[^a-zA-Z]', ' ', str(toxic_comment_test))
toxic_comment_test = toxic_comment_test.lower()

from nltk.stem.porter import PorterStemmer
 
porter_stemmer = PorterStemmer()
toxic_comment_train = porter_stemmer.stem(toxic_comment_train)

toxic_comment_test = porter_stemmer.stem(toxic_comment_test)

from nltk.stem.lancaster import LancasterStemmer

lancaster_stemmer = LancasterStemmer()
l= lancaster_stemmer.stem(toxic_comment_train) 

print (l)

#tokenizing

max_features = 20000

t = Tokenizer(num_words = max_features)
#t.fit_on_texts(word)
t.fit_on_texts(list(toxic_comments_train))


vocab_size = len(t.word_index) + 1
print (vocab_size)

max_length = 200

toxic_tokenized_train = t.texts_to_sequences(toxic_comments_train)
toxic_tokenized_test = t.texts_to_sequences(toxic_comments_test)

x_train = pad_sequences(toxic_tokenized_train, maxlen=max_length,padding='post')

x_test = pad_sequences(toxic_tokenized_test, maxlen = max_length, padding = 'post')



#model_bidirectional lstm
from keras.models import Model
from keras.layers import Input

inp = Input(shape= (max_length, ))
emb_size = 128

x = Embedding(max_features, emb_size)(inp)
x = LSTM(60, return_sequences = True, name ="lstm_layer")(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation = "relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation = "sigmoid")(x)

model = Model(inputs= inp, outputs= x)
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["acc"])

model.fit(x_train, y, batch_size = 32, epochs = 2, validation_split = 0.1)



#model2
model = Sequential()

model.add(Embedding(vocab_size, emb_size, input_length = max_length))
model.add(Flatten())
model.add(LSTM(1000, return_sequences = "True"))
model.add(Dropout(0.1))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(6, activation = 'sigmoid'))

#-------------------------------------------------------------

model.summary()

pred = model.predict(x_test, batch_size = 1024, verbose = 1)

print(pred)

pred.shape

prediction = pd.DataFrame(pred, columns= class_var).to_csv('prediction.csv')