def read_tweets(min_tweets, seed, save_to_file=False):
    import numpy as np
    import pandas as pd
    from collections import defaultdict

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
            id_ = line[2:20]
            text = line[23:-3]
            data_dict[id_] = text
    
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

    # select all lanugages that contain at least 1000 tweets
    selected_langs = groups.query('lang >= {}'.format(min_tweets))
    selected_langs = list(selected_langs.index)

    # remove russian, japanese and arabic
    # selected_langs = [lang for lang in selected_langs if lang not in ['ar', 'ja', 'ru']]
    print(selected_langs, len(selected_langs))

    # select the subset of languages
    tweets = tweets[tweets['lang'].isin(selected_langs)]

    # create the resulting dataframe containing an equal number of entries per language
    result = pd.DataFrame(data=None, columns=tweets.columns)

    for lang in selected_langs:
        words = tweets[tweets['lang'] == lang].sample(n=min_tweets, random_state=seed)
        result = result.append(words)

    print("The resulting shape of the dataset:")
    print(result.shape)

    return result

def preprocess_tweets(data, save_to_file=False):
    import re
    import string
    import html

    # matches any url, or filepath in the logfile   
    url_re = r'https?://[a-zA-Z0-9/.]+'    
    url_re = re.compile(url_re)

    # regular expression pattern matching 2+ whitespace characters
    white_re = r'\s{2,}'
    white_re = re.compile(white_re)

    # hashtag removal
    hash_re = r'#[^\s]+'
    hash_re = re.compile(hash_re)
    
    # @tag removal 
    at_re = r'@[^\s]+'
    at_re = re.compile(at_re)

    # preprocess the data
    for index, row in data.iterrows():
        tweet = row['text']
        tweet = html.unescape(tweet)
        # print(tweet)

        # replace all \n chars with a whitespace
        tweet = tweet.replace("\n", "")
        tweet = tweet.replace("\\n", "")

        # remove all \r chars
        tweet = tweet.replace("\r", "")

        # find all urls
        matches = re.findall(url_re, tweet)

        # replace each url by a single whitespace
        for match in matches:
            tweet = tweet.replace(match, "")

        # # find all hashtags
        # matches = re.findall(hash_re, tweet)

        # # replace each hashtags by a single whitespace
        # for match in matches:
        #     tweet = tweet.replace(match, "")

        # find all @tags
        matches = re.findall(at_re, tweet)

        # replace each @tags by a single whitespace
        for match in matches:
            tweet = tweet.replace(match, "")

        # find all multi-whitespaces
        matches = re.findall(white_re, tweet)

        # replace each multi whitespace by a single whitespace
        for match in matches:
            tweet = tweet.replace(match, "")
        
        # lowercase the tweet
        tweet = tweet.lower()

        transformed_tweet = ""

        for char in tweet:
            if char == " ":
                transformed_tweet += " "
            # remove any numbers in the tweet
            elif str.isdigit(char) or str.isdecimal(char):
                pass
            elif not str.isalpha(char):
                transformed_tweet += '$'
            else:
                transformed_tweet += char
        
        row['text'] = transformed_tweet

    # save the result to file
    if save_to_file:
        data.to_csv("./outputs/preprocessed_tweets.csv", sep=';', header=True, index=False, columns=['lang', 'text'])

    return data