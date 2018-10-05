import re
import numpy as np
import pandas as pd
from collections import defaultdict

# input files
hydrated_json = "./inputs/hydrated.json"
uniformly_sampled = "./inputs/uniformly_sampled.tsv"

# set a seed for random, so results are reproducible
seed = np.random.seed(seed=200)

# dict to store all tweets
tweet_dict = defaultdict(lambda: defaultdict(str))

data_dict = {}

# read hydrated json file
with open(hydrated_json, 'r', encoding='utf-8') as freader:
    data = freader.readlines()

    for line in data: 
        data_dict[line[2:20]] = line[22:-2]
   
hydrated_df = pd.DataFrame(list(data_dict.items()), columns=['id', 'text'])
hydrated_df['id'] = hydrated_df.id.astype(np.int64)
print("Tweets", hydrated_df.shape)

sample_df = pd.read_csv(uniformly_sampled, sep='\t', header=None, names=['lang', 'id'])
print("Language IDs to Tweets", sample_df.shape)

matches = pd.merge(hydrated_df, sample_df, how='left', left_on='id', right_on='id', left_index=True)
print("Matched", matches.shape)

groups = matches.groupby(by=matches['lang']).count()
selected_langs = groups.query('lang > 1000')
print(selected_langs)
selected_langs = list(selected_langs.index)
print(selected_langs)

matches = matches[matches['lang'].isin(selected_langs)]
print("Remaining", matches.shape)

ar = matches[matches['lang'] == 'ar'].sample(n=1000, random_state=seed)
en = matches[matches['lang'] == 'en'].sample(n=1000, random_state=seed)
es = matches[matches['lang'] == 'es'].sample(n=1000, random_state=seed)
fr = matches[matches['lang'] == 'fr'].sample(n=1000, random_state=seed)
ind = matches[matches['lang'] == 'id'].sample(n=1000, random_state=seed)
ja = matches[matches['lang'] == 'ja'].sample(n=1000, random_state=seed)
pt = matches[matches['lang'] == 'pt'].sample(n=1000, random_state=seed)
ru = matches[matches['lang'] == 'ru'].sample(n=1000, random_state=seed)
und = matches[matches['lang'] == 'und'].sample(n=1000, random_state=seed)

result = pd.concat([ar, en, es, fr, ind, ja, pt, ru, und])
print(result)


    # language = url[8:10]
    # doc_id = 'doc_%d' % url_index

    # # replace multiple breaks and spaces by only one space
    # raw = re.sub(pattern, ' ', raw)  

    # # replace every line break with a space
    # raw = re.sub(r'\n', ' ', raw) 

    # # assign each text to its language and lower all uppercase characters
    # tweet_dict[language][doc_id] = raw.lower()  