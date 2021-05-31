## Courtesy of Shuaib's code in sentiment_analysis.py
########### Setting up import library ###########
def import_modules():
    modules = ['inflect', 'contractions', 'plotly', 'emoji', 'emot', 'pandas', 'seaborn', 'matplotlib', 'nltk']
    import importlib
    from tqdm import tqdm
    import os
    for module in tqdm(modules):
        try:
            print(f'Importing {module}')
            importlib.import_module(module)
        except ImportError:
            print(f'Installing {module}, a necessary package for text processing')
            try:
                os.system(f'pip install {module}')
            except:
                print(f'{module} cannot be found using pip. \nRecommedation: Find alternative name of said module and install in command line')
    import nltk as nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

def text_process_pipe(tweet, stopwords_full, num_list, less_freq):
    import re
    import string
    import pandas as pd
    import seaborn as sns
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
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    def replace_contractions(text):
        return contractions.fix(text)

    tweet= replace_contractions(tweet)

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
    # tweet= replace_contractions(tweet)
    # tweet= remove_URL(tweet)
    # tweet = remove_non_ascii(tweet)
    # tweet = replace_numbers(tweet)
    # tweet = lemmatize_verbs(tweet)
    # tweet = deEmojify(tweet)
    # tweet = remove_punctuation(tweet)
    # stopwords_full = list(stopwords.words('english'))
    # commonTwitterStopwords = ['rts','quot', 'sxsw','httpstcoym8csuf43x','rt', 'RT', 'retweet', 'new', 'via', 'us', 'u', 'covid', 'coronavirus', '2019', 'coronavírus',
    #                         '#coronavirus', '19', '#covid', '#covid19','#covid2019', '…', '...', '“', '”', '‘', '’']
    # stopwords_full.extend(commonTwitterStopwords)
    # num_list = '0123456789'
    # freq = pd.Series(' '.join(tweets_df['Content']).split()).value_counts()
    # less_freq = list(freq[freq ==1].index)
                    
    tweet = remove_punct_and_stopwords(tweet, stopwords_full, num_list, less_freq)
    return tweet
