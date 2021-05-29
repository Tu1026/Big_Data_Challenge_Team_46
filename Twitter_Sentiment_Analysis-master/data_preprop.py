import re
import string
import pandas as pd
import seaborn as sns
import nltk
import datetime
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

def replace_contractions(text):
    # """Replace contractions in string of text"""
    return contractions.fix(text)


def remove_URL(sample):
    # """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


def remove_non_ascii(words):
    # """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
        return new_words


def replace_numbers(words):
    #"""Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_punct_and_stopwords(text: str, stopwordlist: list, num_list: list) -> str:

    tknzr = TweetTokenizer()
    try:
        text = remove_URL(text)
        text = replace_contractions(text)
        txt_tokenized = tknzr.tokenize(text)
        text = remove_non_ascii(text)
        text = replace_numbers(text)
        text = ' '.join([char.lower() for char in txt_tokenized if char.lower()
                         not in string.punctuation and char.lower() not in
                         stopwordlist and char not in num_list])
    except TypeError:
            pass

    return text
