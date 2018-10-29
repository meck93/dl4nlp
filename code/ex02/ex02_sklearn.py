import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# set a seed for random, so results are reproducible
seed = np.random.seed(seed=200)

# input file name
input_file = "./input/ted.xml"

# constants
lines = []
talks = {}
relevant = False
talk = ""
key = ""
talk_count = 0

with open(input_file, 'r', encoding='utf-8') as freader:
    lines = freader.readlines()

for line in lines:
    # determine according to the keywords if the talk is relevant
    if '<keywords' in line:
        key = ""

        if 'technology' in line:
            key += "T"
        else:
            key += 'x'          

        if 'entertainment' in line:
            key += "E"
        else:
            key += 'x'  

        if 'design' in line:
            key += 'D'
        else:
            key += 'x'

        if key != 'xxx':
            relevant = True
            continue
        else:
            relevant = False
            continue

    if not relevant:
        continue
    
    # start reading the content
    if '<transcription>' in line:
        talk = ""

    # append each line of the transcript
    elif '<seekvideo' in line:
        start = line.find('>') + 1
        end = line.rfind('<') 
        talk += line[start:end] + " "

    # end of a talk
    elif '</transcription>' in line: 
        # store each talk with key and content
        talks[talk_count] = [key, talk]
        talk_count += 1
        relevant = False

# transform the dict into a dataframe
df = pd.DataFrame.from_dict(talks, orient='index', columns=['label', 'talk'])
df = df.reset_index().drop(columns=['index'])

# show dataframe info
print(df.describe())

# compute the class weight
class_weights = compute_class_weight('balanced', np.unique(df['label']), df['label'])

for label, weight in zip(np.unique(df['label']), class_weights):
    print("label: {} -> weight: {}".format(label, weight))

# split in training and test set
train = df.sample(frac=0.8, random_state=seed)
test = df.drop(train.index)

# for training
y_train = train['label']
x_train = train.drop('label', axis=1)
x_train = x_train['talk'].values

# for testing
y_test = test['label']
x_test = test.drop('label', axis=1)
x_test = x_test['talk'].values

print('Training samples shape: ', x_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test samples shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

# encode the label
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
print(label_encoder.classes_)

# transform the talks
tfvect = TfidfVectorizer(ngram_range=(1,4), max_df=0.99, max_features=3000)
tfvect.fit(x_train)
x_train = tfvect.transform(x_train)
x_test = tfvect.transform(x_test)

# standardize the data
stand = StandardScaler(with_mean=False, with_std=True)
stand.fit(x_train)
x_train = stand.transform(x_train)
x_test = stand.transform(x_test)

# setup base mlp classifier
mlp = MLPClassifier(early_stopping=True, validation_fraction=0.2, random_state=seed, batch_size='auto', 
                    max_iter=200, n_iter_no_change=15, learning_rate='adaptive', verbose=True)

# set up parameter grid to evaluate over
param_grid = dict(hidden_layer_sizes=[(100,100,100), (100,100), (100,)], 
                  solver = ['adam', 'sgd'], activation=['tanh', 'relu'], alpha=[0.001, 0.0001])

# train mlp classifier using randomized grid search
gs_mlp = RandomizedSearchCV(mlp, param_grid, n_iter=5, cv=5, n_jobs=4, verbose=True, refit=True)
gs_mlp.fit(x_train, y_train)

# print the best parameters of the evaluation
print(gs_mlp.best_params_)
print(gs_mlp.best_score_)

# predict the test label
y_pred = gs_mlp.predict(x_test)

# print the accuracy and the confusion matrix
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))

