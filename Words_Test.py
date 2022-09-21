from PIL import Image
import pytesseract as pt
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import words
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import string
string.punctuation
stop = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from english_words import english_words_set

def load_words():
    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words

english_words_1 = load_words()
english_words_1 = list(english_words_1)
english_words_3 = []

for words in english_words_1:
    lemmatized_word = wordnet_lemmatizer.lemmatize(words)
    english_words_3.append(lemmatized_word)

with open(r'/Users/marc/PycharmProjects/Masterarbeit_OCR/english_words_lemmatized.txt', 'w') as fp:
    for item in english_words_3:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

# def lemmatizing(sentence):
#     tokens = word_tokenize(sentence)
#     lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
#     return ' '.join(lemmatized_tokens)
#
# english_words_2 = lemmatizing(english_words_1)
#
#
# text = "last words of every inmate executed since 1984 online HTML table"
#
# count_text = text.apply(lambda x: len([val for val in x.split() if val in english_words_1]))
# print(count_text)