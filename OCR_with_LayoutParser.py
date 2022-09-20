import pdf2image
import numpy as np
import torchvision.ops.boxes as bops
import torch
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
import numpy as np
#import enchant
import cv2
from IPython.display import clear_output
import layoutparser as lp
from datetime import datetime


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
    path = "/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/Testfiles PNG"

    L_prima = []
    title = []

    for imageName in os.listdir(path):
        inputPath = os.path.join(path, imageName)
        if inputPath == path + "/.DS_Store":
            continue

        image = cv2.imread(inputPath)
        title.append(imageName)

        #LayoutParser Modell
        primalayout = lp.Detectron2LayoutModel(config_path='/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/lib/python3.8/site-packages/detectron2/model_zoo/configs/config.yaml', label_map={1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"})

        # Lade Bild in Modell
        prima_layout_result = primalayout.detect(image)
        # Speichere Text Regionen als Text Blocks ab
        prima_text_blocks = lp.Layout([b for b in prima_layout_result if b.type == 'TextRegion'])
        # Initialisiere Tesseract
        ocr_agent = lp.TesseractAgent(languages='eng', config='--psm 11 -c thresholding_method=0')


        for block in prima_text_blocks:
            # Schneide Text Regionen als einzelne Bilder aus Originalbild
            segment_image = (block
                             .pad(left=15, right=15, top=5, bottom=5)
                             .crop_image(image))

            # Führe Tesseract OCR durch
            text = ocr_agent.detect(segment_image)

            # Texte speichern
            block.set(text=text, inplace=True)

        L_test = []
        for txt in prima_text_blocks:
            L_test.append(txt.text)

        string_1 = ""
        for word in L_test:
            string_1 += word + " "

        L_test.clear()
        L_prima.append(string_1)
        L_prima = [w.replace('\n', '') for w in L_prima]



    d = {"Name": title, "Text_prima": L_prima}

    df = pd.DataFrame(d)

    return df


if __name__ == '__main__':
    main()

df = main()
#df["datum"] = df["Name"].str[:19]
#df["datum"] = pd.to_datetime(df["datum"])

def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def lemmatizing(sentence):
    tokens = word_tokenize(sentence)
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


df['Text_processed_prima'] = df['Text_prima'].astype(str).apply(lambda x: remove_punctuation(x))
df['Text_processed_prima'] = df['Text_processed_prima'].apply(lambda x: x.lower())
df['Text_processed_prima'] = df['Text_processed_prima'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Text_processed_prima_token'] = df['Text_processed_prima'].apply(word_tokenize)
df['Text_processed_prima'] = df['Text_processed_prima'].apply(lemmatizing)
df['Text_processed_prima_count'] = df['Text_processed_prima'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))

AnzahlerkannterWörter = df['Text_processed_prima_count'].sum()
print(df['Text_processed_prima_count'].sum())
print(AnzahlerkannterWörter)

#df.to_pickle("Newspaper.pkl")




