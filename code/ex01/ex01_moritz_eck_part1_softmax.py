import re
import numpy as np
import pandas as pd
from collections import defaultdict
import time

def read_raw_and_preprocess(seed):
    # input files
    hydrated_json = "./inputs/hydrated.json"
    uniformly_sampled = "./inputs/uniformly_sampled.tsv"

    # dict to store all tweets
    tweet_dict = defaultdict(lambda: defaultdict(str))

    data_dict = {}

    # read hydrated json file
    with open(hydrated_json, 'r', encoding='utf-8') as freader:
        data = freader.readlines()

        for line in data: 
            data_dict[line[2:20]] = line[22:-2]
    
    raw_text = pd.DataFrame(list(data_dict.items()), columns=['id', 'text'])
    raw_text['id'] = raw_text.id.astype(np.int64)
    print("Tweets", raw_text.shape)

    text_ids = pd.read_csv(uniformly_sampled, sep='\t', header=None, names=['lang', 'id'])
    print("Language IDs to Tweets", text_ids.shape)

    tweets = pd.merge(raw_text, text_ids, how='left', left_on='id', right_on='id', left_index=True)
    print("Matched", tweets.shape)
    print("There exist: {} unique languages".format(len(np.unique(tweets['lang']))))

    # group all tweets by language and count them
    groups = tweets.groupby(by=tweets['lang'], sort=False).count()
    groups = groups.sort_values('lang', ascending=False)
    print("The Top 10 Languages according to tweets")
    print(groups.head(10))

    # select all lanugages that contain at least 1000 tweets
    selected_langs = groups.query('lang >= 1181')
    selected_langs = list(selected_langs.index)

    # remove russian, japanese and arabic
    # selected_langs = [lang for lang in selected_langs if lang not in ['ar', 'ja', 'ru']]
    print(selected_langs, len(selected_langs))

    # select the subset of languages
    tweets = tweets[tweets['lang'].isin(selected_langs)]

    # create the resulting dataframe containing an equal number of entries per language
    result = pd.DataFrame(data=None, columns=tweets.columns)

    for lang in selected_langs:
        words = tweets[tweets['lang'] == lang].sample(n=1181, random_state=seed)
        result = result.append(words)

    print("The resulting shape of the dataset:")
    print(result, result.shape)


    for index, row in result.iterrows():
        tweet = ""
        
        for char in row['text'].lower()[1:-1]: 
            if re.match(r'\s+', char):
                tweet += '_'
            elif not str.isalnum(char):
                tweet += '$'
            else:
                tweet += char
        
        row['text'] = tweet

    # save the result to file
    result.to_csv("./outputs/preprocessed_tweets_smaller.csv", sep=';', header=True, index=False, columns=['lang', 'text'])

def trigrams(text):
    trigrams_dict = dict()
    doc_length = len(text)

    for i in range(0, len(text) - 2):
        trigram = text[i] + text[i+1] + text[i+2]
        
        if trigram not in trigrams_dict:
            trigrams_dict[trigram] = 1
        else:
            trigrams_dict[trigram] += 1
        
    for trigram, count in trigrams_dict.items():
        trigrams_dict[trigram] = count/doc_length

    return trigrams_dict 

def bigrams(text):
    bigrams_dict = dict()
    doc_length = len(text)

    for i in range(0, len(text) - 1):
        bigram = text[i] + text[i+1]
        
        if bigram not in bigrams_dict:
            bigrams_dict[bigram] = 1
        else:
            bigrams_dict[bigram] += 1
        
    for bigram, count in bigrams_dict.items():
        bigrams_dict[bigram] = count/doc_length

    return bigrams_dict  

def count_words(text):
    words_dict = dict()
    doc_length = len(text)

    for word in text:
        if word not in words_dict:
            words_dict[word] = 1
        else:
            words_dict[word] += 1

    for word, count in words_dict.items():
        words_dict[word] = count/doc_length
    
    return words_dict

def normalise_matrix(matrix, mean_and_std=None):
    """
    normalises the data matrix (normalise each datapoint to zero mean and unit variance.)
    :param matrix: input matrix
    :param mean_and_std: provide mean and std as tuples in list for normalisation of test data
    :return: normalised matrix and list consisting of tuples containing mean and std
    """
    normalised = np.ones(matrix.shape)
    
    if mean_and_std == None:
        
        mean_std_list = list()

        for col_index, col in enumerate(matrix):
            mean = matrix[col].mean()
            std = matrix[col].std()
            mean_std_list.append((mean, std))
            for row_index, item in enumerate(matrix[col]):
                try:
                    normalised[row_index, col_index] = (item - mean)/std
                except ZeroDivisionError:
                    normalised[row_index, col_index] = 0.0
        return normalised, mean_std_list
    else:
        for col_index, col in enumerate(matrix):
            for row_index, item in enumerate(matrix[col]):
                try:
                    mean = mean_and_std[col_index][0]
                    std = mean_and_std[col_index][1]
                    normalised[row_index, col_index] = (item - mean)/std
                except ZeroDivisionError:
                    normalised[row_index, col_index] = 0.0
        return normalised

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

        loss = max(0, 1 - (y_value - y_hat_max_value)) + lam * np.linalg.norm(W, 2)
        
        total_loss += loss
        
        y_hat_max_index = np.delete(x.dot(W), y_index).argmax() + 1
        
        if loss > 0:  # not sure whether we need this if statement
            dW[:, y_hat_max_index] += x.transpose()
            dW[:, y_index] -= x.transpose()
        
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
        loss, dW = gradient_reg(X, y, W_learned, -2)
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

    # We use multidimensional array indexing to extract softmax probability of the correct label for each sample.
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
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

def gd_soft_cross(X, y, W, eta, steps):
    """
    Perform gradient descent for a number of times with a fixed learning rate eta
    :param X: data matrix
    :param y: labels
    :param W: weight matrix
    :param eta: learning rate
    :param steps: number of times gradient descent should be performed
    :return: learned representation matrix W_learned
    """  
    for step in range(0, steps):
        loss = cross_entropy(X, y)
        dW = delta_cross_entropy(X, y)
        
        print(loss)

        W = W - eta * dW     

    return W  

# set a seed for random, so results are reproducible
seed = np.random.seed(seed=200)

# preprocess the data (needs to be done only once)
# read_raw_and_preprocess(seed)

# # read preprocessed data from file
# tweets = pd.read_csv("./outputs/preprocessed_tweets_western845.csv", sep=";")

# # bigram storage
# bigrams_full = defaultdict(lambda: defaultdict(dict))

# for index, row in tweets.iterrows():
#     bigrams_dict = bigrams(row['text'])   
#     bigrams_full[row['lang']][index] = bigrams_dict

# # create a new dataframe one row per tweet 
# df = pd.DataFrame()

# start_time = time.time()
# print("Start Time:", start_time)

# for lang, doc in bigrams_full.items():
#     for key, value in doc.items():
#         df = df.append(value, ignore_index=True, verify_integrity=True, sort=False)

# # insert the labels for each tweet
# df.insert(loc=0, column='y', value=tweets['lang'])
# df.to_csv("./outputs/bigrams.csv", sep=";", index=False, na_rep=0)

# read bigrams from csv
data = pd.read_csv("./outputs/bigrams_western845.csv", sep=";")

# split in training and test set
train = data.sample(frac=0.8, random_state=seed)
test = data.drop(train.index)

# for training
y_train = train.y
X_train = train.drop('y', axis=1)

# for testing
y_test = test.y
X_test = test.drop('y', axis=1)

print('Training samples shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test samples shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# normalize the data
train_norm, train_mean_std = normalise_matrix(X_train)
test_norm = normalise_matrix(X_test, train_mean_std)

# transform the y vector to a one-hot encoding
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(np.unique(y_train))

print(np.unique(y_train))
print(lb.classes_)

y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

print(y_train[0])

# append the bias term to X_train
bias_vector = np.ones([train_norm.shape[0], 1])
X_train = np.append(train_norm, bias_vector, axis=1)

# initialise weight matrix with small weights
W = np.random.randn(X_train.shape[1], len(lb.classes_)) * 0.0001

# compute the optimal weights
W_star_reg = gradient_descent_reg(X_train, y_train, W, eta=0.001, steps=25)

# append the bias term to X_test
bias_vector_test = np.ones([X_test.shape[0], 1])
X_test = np.append(test_norm, bias_vector_test, axis=1)

# Compute the number of correctly classified 
true_count = 0
false_count = 0

for index, x in enumerate(X_test.dot(W_star_reg)):
    pred = x.argmax()
    true_label = y_test[index].argmax()

    if pred == true_label:
        true_count += 1
    else:
        false_count += 1

print("Correct Predictions:", true_count)
print("Wrong Predictions:", false_count)
print("Accuracy:", true_count / (true_count + false_count))