import numpy as np
import pandas as pd
import random

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

# set a seed for random, so results are reproducible
seed = np.random.seed(seed=200)

# read preprocessed data from file
tweets = pd.read_csv("./outputs/tweets.csv", sep=";", dtype={'text': str})
print(tweets.head())
print(tweets.describe())

# split in training and test set
train = tweets.sample(frac=0.8, random_state=seed)
test = tweets.drop(train.index)

# for training
y_train = train['lang']
x_train = train.drop('lang', axis=1)
x_train = x_train['text'].values

# for testing
y_test = test['lang']
x_test = test.drop('lang', axis=1)
x_test = x_test['text'].values

print('Training samples shape: ', x_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test samples shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

# encode the label
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# vectorize the data
cvec = CountVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1,3), max_df=1.0, min_df=1, max_features=2500)
cvec.fit(x_train)

x_train = cvec.transform(x_train)
x_test = cvec.transform(x_test)

# apply tfidf
tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
tfidf.fit(x_train)

x_train = tfidf.transform(x_train)
x_test = tfidf.transform(x_test)

# train mlp
mlp = MLPClassifier(early_stopping=True, validation_fraction=0.2, random_state=0, batch_size='auto', verbose=True, max_iter=200, n_iter_no_change=25)

param_grid = dict(hidden_layer_sizes=[(100,), (100,100), (100,100,100), (300,), (300,300)],
                  solver = ['adam'],
                  activation=['tanh', 'relu'], 
                  alpha=[0.001, 0.0001])

gs_mlp = RandomizedSearchCV(mlp, param_grid, n_iter=6, cv=5, n_jobs=4, verbose=1, refit=True)
gs_mlp.fit(x_train, y_train)
print(gs_mlp.best_params_)
print(gs_mlp.best_score_)

# predict the test label
y_pred_svc = gs_mlp.predict(x_test)
print(accuracy_score(y_pred_svc, y_test))