import re
import numpy as np
import pandas as pd
from collections import defaultdict
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from read_transform import read_tweets, preprocess_tweets

def gradient_reg(X, y, W, lam):
    """
    compute the gradient
    :param X: data matrix (train) 
    :param y: the corresponding 
    :param W: weight matrix
    :param lam: reguliser lambda
    :return: Jacobian dW with all gradients
    """
    dW = np.zeros(W.shape)
    
    total_loss = 0.0
    
    for index, x in enumerate(X):
        y_index = y[index].argmax()
        y_value = x.dot(W)[y_index]
        y_hat_max_value = np.delete(x.dot(W), y_index).max()
        loss = max(0, 1 - (y_value - y_hat_max_value))
        total_loss += loss
        y_hat_max_index = np.delete(x.dot(W), y_index).argmax() + 1
        if loss > 0:  # not sure whether we need this if statement
            dW[:, y_hat_max_index] += x.transpose()
            dW[:, y_index] -= x.transpose()
    
    loss /= X.shape[0]
    dW /= X.shape[0]
    
    total_loss += lam * np.linalg.norm(W, 2)
    dW += lam * W
            
    return total_loss, dW

def gradient_descent_reg(X, y, W, eta, steps):
    """
    Perform gradient descent for a number of times with a fixed learning rate eta
    :param X: data matrix
    :param y: labels
    :param W: weight matrix
    :param eta: learning rate
    :param steps: number of times gradient descent should be performed
    :return: learned representation matrix W_learned
    """
    W_learned = W.copy()
    
    for step in range(0, steps):
        loss, dW = gradient_reg(X, y, W_learned, 0.1)
        print(loss)
        W_learned = W_learned - eta * dW
        
    return W_learned

def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    y = y.argmax(axis=1)
    num_samples = y.shape[0]
    p = softmax(X)

    # we use multidimensional array indexing to extract softmax probability of the correct label for each sample.
    log_likelihood = -np.log(p[range(num_samples),y])

    loss = np.sum(log_likelihood) / num_samples   

    return loss

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    y = y.argmax(axis=1)
    num_samples = y.shape[0]

    grad = softmax(X)
    grad[range(num_samples),y] -= 1
    grad = grad/num_samples

    return grad

def gd_soft_cross(X, y, w, eta, steps):
    """
    Perform gradient descent for a number of times with a fixed learning rate eta
    :param X: data matrix
    :param y: labels
    :param w: weight matrix
    :param eta: learning rate
    :param steps: number of times gradient descent should be performed
    :return: learned representation matrix w_learned
    """  
    w_learn = w.copy()

    for step in range(0, steps):
        loss = cross_entropy(X, y)
        dw = delta_cross_entropy(X, y)
        
        print(loss)

        w_learn = w_learn - eta * dw     

    return w_learn

# set a seed for random, so results are reproducible
seed = np.random.seed(seed=200)

# read the data
data = read_tweets(min_tweets=845, seed=seed)

# preprocess the data
data = preprocess_tweets(data=data, save_to_file=True)
data.drop(columns=['id'])
print(data.head(5))

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

# vectorize the tweets
cvec = CountVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1,2), max_df=0.99, min_df=1)
cvec.fit(x_train)

x_train = cvec.transform(x_train)
x_test = cvec.transform(x_test)

# apply tf-idf transformation
transf = TfidfTransformer(use_idf=True, smooth_idf=True)
transf.fit(x_train)
x_train = transf.transform(x_train).toarray()
x_test = transf.transform(x_test).toarray()

# transform the y vector to a one-hot encoding
lb = LabelBinarizer()
lb.fit(np.unique(y_train))

print(np.unique(y_train))
print(lb.classes_)

y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

print(y_train[0])

# append the bias term to x_train
bias_vector = np.ones([x_train.shape[0], 1])
x_train = np.append(x_train, bias_vector, axis=1)

# initialise weight matrix with small weights
w = np.random.randn(x_train.shape[1], len(lb.classes_)) * 0.0001

# compute the optimal weights
w_star_reg = gradient_descent_reg(x_train, y_train, w, eta=0.001, steps=10)
# w_star_reg_cross = gd_soft_cross(x_train, y_train, w, eta=0.001, steps=5)

# append the bias term to x_test
bias_vector_test = np.ones([x_test.shape[0], 1])
x_test = np.append(x_test, bias_vector_test, axis=1)

# Compute the number of correctly classified 
true_count = 0
false_count = 0

for index, x in enumerate(x_test.dot(w_star_reg)):
    pred = x.argmax()
    true_label = y_test[index].argmax()

    if pred == true_label:
        true_count += 1
    else:
        false_count += 1

print("Correct Predictions:", true_count)
print("Wrong Predictions:", false_count)
print("Accuracy:", true_count / (true_count + false_count))