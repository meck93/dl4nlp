import numpy as np
import pandas as pd
import random

from read_transform import read_tweets, preprocess_tweets

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

# set a seed for random, so results are reproducible
seed = np.random.seed(seed=200)

# read the data
data = read_tweets(min_tweets=1181, seed=seed)

# preprocess the data
data = preprocess_tweets(data=data, save_to_file=True)
data.drop(columns=['id'])
print(data.head(5))
print(data.describe())

# split in training and test set
train = data.sample(frac=0.8, random_state=seed)
test = data.drop(train.index)

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
print()

def get_accuracy(y_pred, y_test):
    correct = 0

    for index, prediction in enumerate(y_pred):
        if prediction == y_test[index]:
            correct +=1

    print('Accuracy: ', correct/y_test.shape[0])

# encode the label
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

mnb = Pipeline([
    ('vect', CountVectorizer(strip_accents='unicode', lowercase=True)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('mnb', MultinomialNB())
])

param_grid = dict(vect__ngram_range=[(1,1), (1,2), (1,3), (1,4)], 
                  vect__max_df=[0.99, 0.9], 
                  vect__min_df=[1,2,3],
                  mnb__alpha=[1.0, 0.9, 0.5, 0.1])

nb_gs = RandomizedSearchCV(mnb, param_grid, n_iter=15, cv=5, n_jobs=4, verbose=0, refit=True)
nb_gs.fit(x_train, y_train)
print(nb_gs.best_params_)

# predict the test label
y_pred_nb = nb_gs.predict(x_test)
get_accuracy(y_pred_nb, y_test)

# train a linear support machine machine
svc = Pipeline([
    ('vect', CountVectorizer(strip_accents='unicode', lowercase=True)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('svc', LinearSVC(penalty='l2', random_state=0))
])

param_grid = {'vect__ngram_range':[(1,1), (1,2), (1,3), (1,4)], 
              'vect__max_df':[0.99, 0.9], 
              'vect__min_df':[1,2,3],
              'vect__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
              'svc__loss': ['hinge', 'squared_hinge'],
              'svc__multi_class': ['ovr', 'crammer_singer'],
              'svc__C' : [10.0, 1.0, 0.1, 0.01, 0.001]}

gs_svc = RandomizedSearchCV(svc, param_grid, n_iter=15, cv=5, n_jobs=4, verbose=0, refit=True)
gs_svc.fit(x_train, y_train)
print(gs_svc.best_params_)

# predict the test label
y_pred_svc = gs_svc.predict(x_test)
get_accuracy(y_pred_svc, y_test)

# train a decision tree classifier
tree = Pipeline([
    ('vect', CountVectorizer(strip_accents='unicode', lowercase=True)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('tree', DecisionTreeClassifier(random_state=0))
])

param_grid = {'vect__ngram_range':[(1,1), (1,2), (1,3), (1,4)], 
              'vect__max_df':[0.99, 0.9], 
              'vect__min_df':[1,2,3],
              'vect__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
              "tree__max_depth": [6, 12, None],
              "tree__min_samples_leaf": np.random.randint(1, 9, size=3),
              "tree__criterion": ["gini", "entropy"]}

gs_tree = RandomizedSearchCV(tree, param_grid, n_iter=15, cv=5, n_jobs=4, verbose=0, refit=True)
gs_tree.fit(x_train, y_train)
print(gs_tree.best_params_)

# predict the test label
y_pred_svc = gs_tree.predict(x_test)
print(accuracy_score(y_pred_svc, y_test))
