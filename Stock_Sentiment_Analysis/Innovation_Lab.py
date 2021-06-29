
import re
import nltk
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn import metrics
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor,LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import random
import itertools
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import loguniform
import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('ggplot')
import tensorflow
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense,Dropout, Activation, Lambda, Embedding, LSTM, SimpleRNN, GRU,Convolution1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
stop_words=nltk.corpus.stopwords.words('english')
stop_words.append('b')
stop_words.append('nan')
from nltk.tokenize import word_tokenize
nltk.download('punkt')

#reading data
data=pd.read_csv("data.csv",encoding='ISO-8859-1')
x=data
y=data['Label']

#combining news headlines 
T=x.iloc[:,2:]
T.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
headlines = []
for row in range(0,len(T.index)):
    headlines.append(' '.join(str(x) for x in T.iloc[row,0:]))
for i in range(len(headlines)):
    headlines[i]=headlines[i].lower()
for i in range(len(headlines)):
    sr=headlines[i]
    tk=word_tokenize(sr)
    token_wsw=[word for word in tk if not word in stop_words]
    sen = (" ").join(token_wsw)
    headlines[i]=sen

arr=pd.Series(headlines)
df=pd.concat([arr,y],axis=1)
df.rename(columns = {0:'headlines'}, inplace = True)

#train-test split
x_train1,x_test1,y_train,y_test=train_test_split(headlines,y,random_state=42,test_size=0.3)

#using TFidfVectorizer
vector=TfidfVectorizer(min_df=0.02,max_df=0.3,max_features=100000,ngram_range=(2,2))
# # tf=TfidfTransformer()
# vector=CountVectorizer()
x_train=vector.fit_transform(x_train1)
x_test=vector.transform(x_test1)
# x_train=tf.fit_transform(x_train);
# x_test=tf.transform(x_test)

#logistic regression
lr=LogisticRegression(random_state=42)
lr.fit(x_train,y_train)
prediction=lr.predict(x_test)
score=accuracy_score(y_test,prediction)
print("Logistic Regression")
print(score)
print(confusion_matrix(y_test,prediction))

#naive bayes
mnb=MultinomialNB(alpha=0.2)
mnb.fit(x_train,y_train)
prediction=mnb.predict(x_test)
score=accuracy_score(y_test,prediction)
print("\n")
print("Multinomial Naive Bayes")
print(score)
print(confusion_matrix(y_test,prediction))

#random forest
rf=RandomForestClassifier(random_state=42,max_depth=100,n_estimators=200)
rf.fit(x_train,y_train)
prediction=rf.predict(x_test)
score=accuracy_score(y_test,prediction)
print("\n")
print("Random Forest")
print(score)
print(confusion_matrix(y_test,prediction))

#xgboost
xgb=XGBClassifier(random_state=42)
xgb.fit(x_train,y_train)
prediction=xgb.predict(x_test)
score=accuracy_score(y_test,prediction)
print("\n")
print("XGB")
print(score)
print(confusion_matrix(y_test,prediction))

gbm=GradientBoostingClassifier(random_state=45,n_estimators=100,learning_rate=0.2,max_depth=3)
gbm.fit(x_train,y_train)
prediction=gbm.predict(x_test)
score=accuracy_score(y_test,prediction)
print("\n")
print("GBM")
print(score)
print(confusion_matrix(y_test,prediction))

#Deep Learning Models
#Multi Level Perceptron
from sklearn.preprocessing import RobustScaler, StandardScaler
y_train1 = to_categorical(y_train,2)
y_test1 = to_categorical(y_test,2)
x_train=x_train.toarray()
x_test=x_test.toarray()
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_train=sc.fit_transform(x_train)
input_dim=x_train.shape[1]
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(Dense(256, input_dim=input_dim,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu',))
model.add(Dropout(0.4))
model.add(Dense(2,activation='sigmoid'))
#opt=SGD()
opt=Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
history=model.fit(x_train, y_train1, epochs=25, batch_size=32,validation_data=(x_test,y_test1))
preds = model.predict_classes(x_test)
acc = accuracy_score(y_test, preds)
train_acc=model.evaluate(x_train, y_train1, verbose=0)
test_acc=model.evaluate(x_test, y_test1, verbose=0)
print(train_acc)
print(test_acc)
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

print("Multi Level Perceptron")
print(acc)
print(confusion_matrix(y_test,preds))

#LSTM
print("LSTM")
max_features = 100000
EMBEDDING_DIM = 100
maxlen = 100
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train1)
sequences_train = tokenizer.texts_to_sequences(x_train1)
sequences_test = tokenizer.texts_to_sequences(x_test1)
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)
Y_train = to_categorical(y_train,2)
Y_test = to_categorical(y_test,2)
model_lstm=Sequential()
model_lstm.add(Embedding(max_features, 256))
model_lstm.add(Dropout(0.4))
model_lstm.add(LSTM(128))
model_lstm.add(Dense(2,activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model_lstm.summary()
history=model_lstm.fit(X_train, Y_train, batch_size=16, epochs=2,validation_data=(X_test, Y_test))
score, acc = model_lstm.evaluate(X_test, Y_test,batch_size=16)
print('Test score:', score)
print('Test accuracy:', acc)
print("Generating test predictions...")
preds = model_lstm.predict_classes(X_test)
acc = accuracy_score(y_test,preds)
train_acc=model_lstm.evaluate(X_train, Y_train, verbose=0)
test_acc=model_lstm.evaluate(X_test, Y_test, verbose=0)
print(train_acc)
print(test_acc)
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
print("\n")
plt.show()



