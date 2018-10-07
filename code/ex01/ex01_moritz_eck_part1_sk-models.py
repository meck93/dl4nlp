import numpy as np
import pandas as pd
import random

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

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

nb_clf = Pipeline([
    ('vect', CountVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1,3), max_df=1.0, min_df=1)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('nb_clf', MultinomialNB())
])

# train the multinomial naive bayes model
nb_clf.fit(x_train, y_train)
scores = cross_val_score(nb_clf, x_train, y_train, scoring='accuracy', cv=10)
print(scores)

# predict the test label
y_pred_nb = nb_clf.predict(x_test)
get_accuracy(y_pred_nb, y_test)

# train a linear support machine machine
svc = Pipeline([
    ('vect', CountVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1,3), max_df=1.0, min_df=1)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('svc', LinearSVC(penalty='l2', random_state=0))
])

param_grid = {'svc__loss': ['hinge', 'squared_hinge'],
             'svc__multi_class': ['ovr', 'crammer_singer'],
             'svc__C' : [10.0, 1.0, 0.1, 0.01]}

gs_svc = GridSearchCV(svc, param_grid, cv=5, n_jobs=4, verbose=0)
gs_svc.fit(x_train, y_train)
print(gs_svc.best_params_)

# predict the test label
y_pred_svc = gs_svc.predict(x_test)
get_accuracy(y_pred_svc, y_test)

# train a decision tree classifier
tree = Pipeline([
    ('vect', CountVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1,3), max_df=1.0, min_df=1)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('tree', DecisionTreeClassifier(random_state=0))
])

param_grid = {"tree__max_depth": [6, 12, None],
              "tree__min_samples_leaf": np.random.randint(1, 9, size=3),
              "tree__criterion": ["gini", "entropy"]}

gs_tree = GridSearchCV(tree, param_grid, cv=5, n_jobs=4, verbose=0)
gs_tree.fit(x_train, y_train)
print(gs_tree.best_params_)

# predict the test label
y_pred_svc = gs_tree.predict(x_test)
print(accuracy_score(y_pred_svc, y_test))

