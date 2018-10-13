import re
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder

from read_transform import read_tweets, preprocess_tweets

from sklearn.metrics import accuracy_score

def multi_class_hing_loss(W, X, y, reg):
  # initialize the gradient as zero
  dW = np.zeros(W.shape)
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0

  for i in range(num_train):
    scores = X[i,:].dot(W)

    # use argmax since we used one-hot encoding for the label
    y_ = y[i].argmax()

    correct_class_score = scores[y_]

    for j in range(num_classes):
      if j == y_:
        continue

      margin = scores[j] - correct_class_score + 1 

      if margin > 0:
        loss += margin
        dW[:,y_] -= X[i,:] 
        dW[:,j] += X[i,:] 

  # Averaging over all examples
  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  
  return loss, dW

def gradient_descent(X, y, w, eta, steps):
    w_learn = w.copy()

    for step in range(0, steps):
        loss, dW = multi_class_hing_loss(w_learn, X, y, 0.5)
        print("current loss:", loss)
        w_learn = w_learn - eta * dW     

    return w_learn

class Softmax(object):    

  def __init__(self):
    self.W = None
    self.b = None

  def softmax(self, X):
    # Normalize the scores beforehand with max as zero to avoid computational problems with the exponential
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    # Softmax activation
    return exps / np.sum(exps, axis=1, keepdims=True)
    
  def get_loss_grads(self, X, y, reg, n_features, n_samples, n_classes):
    # Linear mapping scores
    scores = np.dot(X, self.W)+self.b

    # Softmax activation
    probs = self.softmax(scores)

    # Logloss of the correct class for each of our samples
    correct_logprobs = -np.log(probs[np.arange(n_samples), y])

    # Compute the average loss
    loss = np.sum(correct_logprobs)/n_samples

    # Add regularization using the L2 norm
    reg_loss = 0.5*reg*np.sum(self.W*self.W)
    loss += reg_loss
    
    # Gradient of the loss with respect to scores
    dscores = probs.copy()

    # Substract 1 from the scores of the correct class
    dscores[np.arange(n_samples),y] -= 1
    dscores /= n_samples

    # Gradient of the loss with respect to weights
    dW = X.T.dot(dscores) 

    # Add gradient regularization 
    dW += reg*self.W

    # Gradient of the loss with respect to biases
    db = np.sum(dscores, axis=0, keepdims=True)

    return loss, dW, db

  def train(self, X, y, learning_rate=1e-4, reg=0.5, num_iters=500):       
    # Get useful parameters
    n_features, n_samples = X.shape[1], X.shape[0]   
    n_classes = len(np.unique(y))
    
    # Initialize weights from a normal distribution and the biases with zeros
    if (self.W is None) & (self.b is None):
      self.W = np.random.randn(x_train.shape[1], n_classes) * 0.0001
      self.b = np.zeros((1, n_classes))
        
    for iter in range(num_iters):
      # Get loss and gradients
      loss, dW, db = self.get_loss_grads(X, y, reg, n_features, n_samples, n_classes)
      print("loss:", loss)
      
      # update weights and biases
      self.W -= learning_rate*dW
      self.b -= learning_rate*db 

  def predict(self, X):
    y_pred = np.dot(X, self.W)+self.b
    y_pred=np.argmax(y_pred, axis=1)
    
    return y_pred

# set a seed for random, so results are reproducible
seed = np.random.seed(seed=200)

# read the data
data = read_tweets(min_tweets=845, seed=seed)

# preprocess the data
data = preprocess_tweets(data=data, save_to_file=False)
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
cvec = CountVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1,2), max_df=0.99, min_df=1, max_features=500)
cvec.fit(x_train)
x_train = cvec.transform(x_train)
x_test = cvec.transform(x_test)

# apply tf-idf transformation
transf = TfidfTransformer(use_idf=True, smooth_idf=True)
transf.fit(x_train)
x_train = transf.transform(x_train).toarray()
x_test = transf.transform(x_test).toarray()
print(x_train.shape)

# encode the y vector from 0 to num_classes
le = LabelEncoder()
le.fit(np.unique(y_train))
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)

# transform the y vector to a one-hot encoding
lb = LabelBinarizer()
lb.fit(np.unique(y_train))
y_train_bin = lb.transform(y_train)
y_test_bin = lb.transform(y_test)

# number of classes
n_classes = len(lb.classes_)
print(lb.classes_, n_classes)

# append the bias term to x_train
bias_vector = np.ones([x_train.shape[0], 1])
x_train = np.append(x_train, bias_vector, axis=1)

# initialise weight matrix with small weights
w = np.random.randn(x_train.shape[1], len(lb.classes_)) * 0.0001

# compute the optimal weights
w_star_reg = gradient_descent(x_train, y_train_bin, w, eta=0.01, steps=15)

# append the bias term to x_test
bias_vector_test = np.ones([x_test.shape[0], 1])
x_test = np.append(x_test, bias_vector_test, axis=1)

# Compute the number of correctly classified 
true_count = 0
false_count = 0

for index, x in enumerate(x_test.dot(w_star_reg)):
    pred = x.argmax()
    true_label = y_test_bin[index].argmax()

    if pred == true_label:
        true_count += 1
    else:
        false_count += 1

print("Correct Predictions:", true_count)
print("Wrong Predictions:", false_count)
print("Accuracy:", true_count / (true_count + false_count))

# softmax implementation 
softmax = Softmax()
print(y_train_encoded.shape)
print(x_train.shape)

# train the model using softmax + corss entropy loss
softmax.train(x_train, y_train_encoded, learning_rate=0.001, reg=0.5, num_iters=200)

# predict the labels 
pred = softmax.predict(x_test)

# evaluate the result (compute the accuracy)
acc = accuracy_score(pred, y_test_encoded)
print(acc)