from PIL import Image
import pytesseract as pt
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import words
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import string

string.punctuation
stop = stopwords.words('english')
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
from english_words import english_words_set
import numpy as np
# import enchant
import cv2
from pythonRLSA import rlsa
import math

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 10000)


def load_words():
    with open('english_words_lemmatized.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words

english_words_1 = load_words()


def main():
    # Pfad um Ordner mit Bildern zu bekommen
    path = "/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/Testfiles"

    L = []
    title = []

    # Iteriere über jedes Bild im Ordner
    for imageName in os.listdir(path):
        inputPath = os.path.join(path, imageName)
        if inputPath == path + "/.DS_Store":
            continue
        img = Image.open(inputPath)

        image = cv2.imread(inputPath)
        # image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert2grayscale
        (thresh, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # convert2binary
        cv2.imshow('gray', gray)
        # cv2.imwrite('binary.png', binary)

        # OCR anwenden und in Liste L anfügen
        text = pt.image_to_string(image, lang="eng")
        L.append(text)
        title.append(imageName)

        # Entferne '\n' für Absätze in L
        L = [w.replace('\n', '') for w in L]

    d = {"Name": title, "Text": L}
    df = pd.DataFrame(d)
    return df


if __name__ == '__main__':
    main()

df = main()


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def stem_sentences(sentence):
    tokens = word_tokenize(sentence)
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def lemmatizing(sentence):
    tokens = word_tokenize(sentence)
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


def process_words(df):
    df['Text_processed'] = df['Text'].astype(str).apply(lambda x: remove_punctuation(x))
    df['Text_processed'] = df['Text_processed'].apply(lambda x: x.lower())
    df['overview_without_stopwords'] = df['Text_processed'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df["token"] = df["overview_without_stopwords"].apply(word_tokenize)
    df["token_stemmed"] = df["overview_without_stopwords"].apply(stem_sentences)
    df["token_lemmatized"] = df["overview_without_stopwords"].apply(lemmatizing)
    df["token_lemmatized_stemmed"] = df["token_lemmatized"].apply(stem_sentences)
    df['count'] = df['overview_without_stopwords'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
    df['count_english_words'] = df['overview_without_stopwords'].apply(lambda x: len([val for val in x.split() if val in english_words_set]))
    df['count_stemmed'] = df['token_stemmed'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
    df['count_english_words_stemmed'] = df['token_stemmed'].apply(lambda x: len([val for val in x.split() if val in english_words_set]))
    df['count_lemmatized'] = df['token_lemmatized'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
    df['count_english_words_lemmatized'] = df['token_lemmatized'].apply(lambda x: len([val for val in x.split() if val in english_words_set]))
    df['count_lemmatized_stemmed'] = df['token_lemmatized_stemmed'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
    df['count_english_words_lemmatized_stemmed'] = df['token_lemmatized_stemmed'].apply(lambda x: len([val for val in x.split() if val in english_words_set]))
    return df


df = process_words(df)

sizes = df["count"]
labels = df["Name"]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig("Grayscale_Kuchen.png")
plt.show()


yvalues = [df["count"].sum(), df["count_stemmed"].sum(), df["count_lemmatized"].sum(), df["count_lemmatized_stemmed"].sum()]
xvalues = ["Roh", "Stemmed", "Lemmatisiert", "Stemmed und Lemmatisiert"]
plt.bar(xvalues, yvalues)
plt.xlabel("Methode")
plt.ylabel("Anzahl erkannter Wörter")
plt.savefig("Image_Bars.png")
plt.show()

print(df)
# df.plot.pie(y='count', figsize=(5, 5))
# plt.savefig("Kuchen.png")