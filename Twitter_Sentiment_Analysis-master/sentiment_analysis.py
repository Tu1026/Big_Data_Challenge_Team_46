# %pip install contractions
# %pip install inflect
# %pip install plotly
# %pip install emoji
# %pip install emot

import os
import re
import json
import pickle
import string
import pandas as pd
import seaborn as sns
import nltk
import datetime
import wordcloud
import matplotlib.pyplot as plt
import re
import unicodedata
import contractions
import inflect
import matplotlib.pyplot as plt
import seaborn as sns
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from googletrans import Translator
from nltk.stem import PorterStemmer, WordNetLemmatizer
# import data_preprop as pp

############ Data processing #################

df = pd.read_csv('Sentiment_202105272020.csv')
tweets_df=df

tweets_df.shape[0]
tweets_df = tweets_df[tweets_df['Content'].notnull()].reset_index()
tweets_df.drop(columns=['index','Logits_Neutral','Logits_Positive','Logits_Negative'], inplace=True)
tweets_df.shape[0]

tweets_df['Created_at']=pd.to_datetime(tweets_df['Created_at'],utc=True)
tweets_df
tweets_df['Year']=pd.DatetimeIndex(tweets_df['Created_at']).year
tweets_df['Month'] = pd.DatetimeIndex(tweets_df['Created_at']).month
tweets_df


def replace_contractions(text):
    return contractions.fix(text)

tweets_df['Tweets_Clean'] = tweets_df['Content'].apply(lambda x: replace_contractions(x))

def remove_URL(sample):
    return re.sub(r"http\S+", "", sample)

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def replace_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def stem_words(words):
    stemmer = PorterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def deEmojify(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
                             
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
# Function for removing emoticons
def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words
                             
                            
# def tweet_cleaner(tweet):
#     '''
#     Function to remove punctuations, special characters, html links, twitter handels etc...
#     '''
    
#     stopwords = ['rt','rts', 'retweet', 'quot', 'sxsw']
    
#     punctuation = set(string.punctuation) # punctuation of English language
#     punctuation.remove('#') # remove # so hashtags remain in x
    
#     x = tweet
#     x = re.sub(r'https?:\/\/\S+', '', x) # remove URL references
#     x = re.sub(r'{link}', '', x)  # remove placeholders
#     x = re.sub(r'@[\w]*', '', x) # remove @mention users
#     x = re.sub('[^A-Za-z0-9]+', ' ', x) # remove @mention users
#     x = re.sub(r'\b[0-9]+\b', '', x) # remove stand-alone numbers  
#     x = re.sub(r'&[a-z]+;', '', x) # remove HTML reference characters
#     x = ''.join(ch for ch in x if ch not in punctuation) # remove punctuation
#     x = x.replace("[^a-zA-z#]", " ")  #remove special characters

#     x = [word.lower() for word in x.split() if word.lower() not in stopwords]
#     x = [w for w in x if len(w)>2]

#     return ' '.join(x)


def remove_punct_and_stopwords(text: str, stopwordlist: list, num_list: list, less_freq: list) -> str:
    try:
        text = remove_URL(text)
        text = replace_contractions(text)
        text = deEmojify(text)
        text = remove_emoticons(text)
        tknzr = TweetTokenizer()
        txt_tokenized = tknzr.tokenize(text)
        txt_tokenized = stem_words(txt_tokenized)
        txt_tokenized = lemmatize_verbs(txt_tokenized)
        txt_tokenized = remove_non_ascii(txt_tokenized)
        txt_tokenized = replace_numbers(txt_tokenized)
        txt_tokenized = remove_punctuation(txt_tokenized)
        text = ' '.join([char.lower() for char in txt_tokenized if char.lower() 
                         not in string.punctuation and char.lower() not in
                         stopwordlist and char not in num_list and char not in less_freq])
    except TypeError:
       pass
   
    return text

# def remove_punct_and_stopwords(text: str, stopwordlist: list, num_list: list) -> str:

#     tknzr = TweetTokenizer()
#     try:
#         text = remove_URL(text)
#         text = replace_contractions(text)
#         txt_tokenized = tknzr.tokenize(text)
#         text = ' '.join([char.lower() for char in txt_tokenized if char.lower() 
#                          not in string.punctuation and char.lower() not in
#                          stopwordlist and char not in num_list])
#     except TypeError:
#        pass
   
#     return text

stopwords_full = list(stopwords.words('english'))
commonTwitterStopwords = ['rts','quot', 'sxsw','httpstcoym8csuf43x','rt', 'RT', 'retweet', 'new', 'via', 'us', 'u', 'covid', 'coronavirus', '2019', 'coronavírus',
                          '#coronavirus', '19', '#covid', '#covid19','#covid2019', '…', '...', '“', '”', '‘', '’']

stopwords_full.extend(commonTwitterStopwords)
num_list = '0123456789'
freq = pd.Series(' '.join(tweets_df['Content']).split()).value_counts()
less_freq = list(freq[freq ==1].index)


tweets_df['Tweets_Clean'] = tweets_df['Content'].apply(lambda x: remove_punct_and_stopwords(x, stopwords_full, num_list,less_freq ))
tweets_df

plt.figure(figsize=(14, 8))
GB=tweets_df.groupby(['Month','Year'])['Tweets_Clean'].count().reset_index()
bar_plot = sns.barplot(x='Month', y='Tweets_Clean', data=GB)
for p in bar_plot.patches:
             bar_plot.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=13, color='black', xytext=(0, 6), textcoords='offset points')
plt.title('Number of Tweets per Month', fontweight='bold')
plt.grid(True, alpha=0.1, c='black')
plt.ylabel('Tweet Count')
plt.xlabel('Months/Year')
plt.show()
plt.savefig('tweet_count_trend.png')

sns.countplot(x = 'Month', hue='Sentiment_Label', 
            palette = 'GnBu',
            data=tweets_df)
plt.title('Setiments over time', fontweight='bold')
plt.show()
plt.savefig('sentiment_trend.png')


############ Data exploration ######################

from sentiment_class import TwitterSentiment, month_as_string

tweets_all_months = TwitterSentiment(input_df=tweets_df, tweet_column='Tweets_Clean')
tweets_all_months.plot_most_common_words(n_most_common=15, figsize=(10, 8))
tweets_all_months.plot_wordcloud(figsize=(10, 8))
                                                       
# def bigram_builder(text):
#   """
#   A simple function to clean up the data. All the words that
#   are not designated as a stop word is then lemmatized after
#   encoding and basic regex parsing are performed.
#   """
#   wnl = nltk.stem.WordNetLemmatizer()
#   stopwords = stopwords_full 
#   text = (unicodedata.normalize('NFKD', text)
#     .encode('ascii', 'ignore')
#     .decode('utf-8', 'ignore')
#     .lower())
#   words = re.sub(r'[^\w\s]', '', text).split()
#   return [wnl.lemmatize(word) for word in words if word not in stopwords]

# tweets_df['Tweets_Clean'] = tweets_df['Content'].apply(lambda x: bigram_builder(x))
# tweets_df
# words = bigram_builder(''.join(str(tweets_df['Tweets_Clean'].tolist())))
# words
# (pd.Series(nltk.ngrams(words, 2)).value_counts())[:10]
# bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]
# bigrams_series.sort_values().plot.barh(colormap='Accent', width=.9, figsize=(12, 8))

from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(4,4)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words2 = get_top_n_bigram(tweets_df['Tweets_Clean'], 20)
df2 = pd.DataFrame(common_words2,columns=['Tweet', "Count"])
df2.head()

df2.groupby('Tweet').sum()['Count'].sort_values().plot(
    kind='barh',
    figsize=(12,6),
    xlabel = "4-gram Words",
    colormap='Accent'
)
plt.title('20 Most Frequently Occuring Bigrams')
plt.ylabel('4-gram')
plt.xlabel('# of Occurrences')
plt.savefig('bigram.png')
plt.show()
plt.close()
                                                       
my_colors = ['red', 'orange', 'green'] 
df['Sentiment_Label'].value_counts().plot(kind='bar',color=my_colors)
plt.title('Tweet sentiments')
plt.ylabel('Counts')
plt.xlabel('Emotion')
plt.show()
plt.savefig('sentiment_bar.png')




