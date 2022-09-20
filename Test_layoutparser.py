import pdf2image
import numpy as np
import torchvision.ops.boxes as bops
import torch
import cv2
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
import spacy
from IPython.display import clear_output
from skimage.filters import (threshold_niblack, threshold_sauvola)
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
    path = "/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/Testfile_1"

    L_prima = []
    L = []
    #L_gray = []
    #L_otsu = []
    # L_otsu_inverse = []
    #L_niblack = []
    #L_sauvola = []
    title = []


    for imageName in os.listdir(path):
        inputPath = os.path.join(path, imageName)
        if inputPath == path + "/.DS_Store":
            continue



        image = cv2.imread(inputPath)
        # image = cv2.imread(img)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert2grayscale
        #(thresh, binary) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # convert2binary
        #(thresh, binary_inverse) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #thresh_niblack = threshold_niblack(gray)
        #img_niblack = gray > thresh_niblack
        #thresh_sauvola = threshold_sauvola(gray)
        #img_sauvola = gray > thresh_sauvola

        text = pt.image_to_string(image, lang="eng", config='--psm 11 -c thresholding_method=0')
        #text_gray = pt.image_to_string(gray, lang="eng")
        #text_otsu = pt.image_to_string(binary, lang="eng")
        #text_otsu_inverse = pt.image_to_string(binary_inverse, lang="eng")
        #text_niblack = pt.image_to_string(img_niblack, lang="eng")
        #text_sauvola = pt.image_to_string(img_sauvola, lang="eng")

        L.append(text)
        #L_gray.append(text_gray)
        #L_otsu.append(text_otsu)
        #L_otsu_inverse.append(text_otsu_inverse)
        #L_otsu_inverse.append(L_otsu)
        #L_niblack.append(text_niblack)
        #L_sauvola.append(text_sauvola)
        title.append(imageName)

        L = [w.replace('\n', '') for w in L]
        #L_gray = [w.replace('\n', '') for w in L_gray]
        #L_otsu = [w.replace('\n', '') for w in L_otsu]
        #L_otsu_inverse = [w.replace('\n', '') for w in L_otsu_inverse]
        #L_niblack = [w.replace('\n', '') for w in L_niblack]
        #L_sauvola = [w.replace('\n', '') for w in L_sauvola]


        primalayout = lp.Detectron2LayoutModel(config_path='/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/lib/python3.8/site-packages/detectron2/model_zoo/configs/config.yaml', label_map={1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"})

        prima_layout_result = primalayout.detect(image)
        prima_text_blocks = lp.Layout([b for b in prima_layout_result if b.type == 'TextRegion'])

        ocr_agent = lp.TesseractAgent(languages='eng', config='--psm 11 -c thresholding_method=0')



        for block in prima_text_blocks:
            # Crop image around the detected layout
            segment_image = (block
                             .pad(left=15, right=15, top=5, bottom=5)
                             .crop_image(image))

            # Perform OCR
            text = ocr_agent.detect(segment_image)
            #text_test = pt.image_to_string(segment_image, lang="eng", config='--psm 11 -c thresholding_method=0')

            # Save OCR result
            block.set(text=text, inplace=True)

        L_test = []
        for txt in prima_text_blocks:
            L_test.append(txt.text)

        #L_prima = " ".join(L_prima)
        string_1 = ""
        for word in L_test:
            string_1 += word + " "

        L_test.clear()
        L_prima.append(string_1)
        L_prima = [w.replace('\n', '') for w in L_prima]
        #string_1 = string_1.replace('\n', '')
        #print(string_1)


    #d = {"Name": title, "Text": L, "Text_gray": L_gray, "Text_otsu": L_otsu, "Text_otsu_inverse": L_otsu_inverse,
    #     "Text_niblack": L_niblack, "Text_sauvola": L_sauvola}
    d = {"Name": title, "Text": L}#, "Text_gray": L_gray}#, "Text_otsu": L_otsu, "Text_niblack": L_niblack, "Text_sauvola": L_sauvola}
    #d2 = pd.Series(L_prima)#, "Text_prima": L_prima}
    d2 = {"Text_prima": L_prima}

    df = pd.DataFrame(d)
    #df2 = d2.to_frame()
    df2 = pd.DataFrame(d2)
    print(df2)

    df = pd.concat([df, df2], axis=1)
    #df.rename(columns={0: "Text_prima"})


    return df


if __name__ == '__main__':
    main()

df = main()
df["datum"] = df["Name"].str[:19]
df["datum"] = pd.to_datetime(df["datum"])

def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def lemmatizing(sentence):
    tokens = word_tokenize(sentence)
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

df['Text_processed'] = df['Text'].astype(str).apply(lambda x: remove_punctuation(x))
#df['Text_processed_gray'] = df['Text_gray'].astype(str).apply(lambda x: remove_punctuation(x))
#df['Text_processed_otsu'] = df['Text_otsu'].astype(str).apply(lambda x: remove_punctuation(x))
#df['Text_processed_otsu_inverse'] = df['Text_otsu_inverse'].astype(str).apply(lambda x: remove_punctuation(x))
#df['Text_processed_niblack'] = df['Text_niblack'].astype(str).apply(lambda x: remove_punctuation(x))
#df['Text_processed_sauvola'] = df['Text_sauvola'].astype(str).apply(lambda x: remove_punctuation(x))
df['Text_processed_prima'] = df['Text_prima'].astype(str).apply(lambda x: remove_punctuation(x))

df['Text_processed'] = df['Text_processed'].apply(lambda x: x.lower())
#df['Text_processed_gray'] = df['Text_processed_gray'].apply(lambda x: x.lower())
#df['Text_processed_otsu'] = df['Text_processed_otsu'].apply(lambda x: x.lower())
#df['Text_processed_otsu_inverse'] = df['Text_processed_otsu_inverse'].apply(lambda x: x.lower())
#df['Text_processed_niblack'] = df['Text_processed_niblack'].apply(lambda x: x.lower())
#df['Text_processed_sauvola'] = df['Text_processed_sauvola'].apply(lambda x: x.lower())
df['Text_processed_prima'] = df['Text_processed_prima'].apply(lambda x: x.lower())

df['Text_processed'] = df['Text_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df['Text_processed_gray'] = df['Text_processed_gray'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df['Text_processed_otsu'] = df['Text_processed_otsu'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df['Text_processed_otsu_inverse'] = df['Text_processed_otsu_inverse'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df['Text_processed_niblack'] = df['Text_processed_niblack'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df['Text_processed_sauvola'] = df['Text_processed_sauvola'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Text_processed_prima'] = df['Text_processed_prima'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df['Text_processed_token'] = df['Text_processed'].apply(word_tokenize)
#df['Text_processed_gray_token'] = df['Text_processed_gray'].apply(word_tokenize)
#df['Text_processed_otsu_token'] = df['Text_processed_otsu'].apply(word_tokenize)
#df['Text_processed_otsu_token_inverse'] = df['Text_processed_otsu_inverse'].apply(word_tokenize)
#df['Text_processed_niblack_token'] = df['Text_processed_niblack'].apply(word_tokenize)
#df['Text_processed_sauvola_token'] = df['Text_processed_sauvola'].apply(word_tokenize)
df['Text_processed_prima_token'] = df['Text_processed_prima'].apply(word_tokenize)

df['Text_processed'] = df['Text_processed'].apply(lemmatizing)
#df['Text_processed_gray'] = df['Text_processed_gray'].apply(lemmatizing)
#df['Text_processed_otsu'] = df['Text_processed_otsu'].apply(lemmatizing)
#df['Text_processed_otsu_inverse'] = df['Text_processed_otsu_inverse'].apply(lemmatizing)
#df['Text_processed_niblack'] = df['Text_processed_niblack'].apply(lemmatizing)
#df['Text_processed_sauvola'] = df['Text_processed_sauvola'].apply(lemmatizing)
df['Text_processed_prima'] = df['Text_processed_prima'].apply(lemmatizing)

df['Text_processed_count'] = df['Text_processed'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
#df['Text_processed_gray_count'] = df['Text_processed_gray'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
#df['Text_processed_otsu_count'] = df['Text_processed_otsu'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
#df['Text_processed_otsu_count_inverse'] = df['Text_processed_otsu_inverse'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
#df['Text_processed_niblack_count'] = df['Text_processed_niblack'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
#df['Text_processed_sauvola_count'] = df['Text_processed_sauvola'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
df['Text_processed_prima_count'] = df['Text_processed_prima'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))

#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text', 'Text_gray', 'Text_otsu', 'Text_niblack', 'Text_sauvola']].astype(str).apply(lambda x: lower(x))
#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']].apply(lambda x: x.lower())
#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']].apply(word_tokenize)
#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']].apply(lemmatizing)


#
# sizes = df["Text_processed_count"]
# labels = df["Name"]
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.savefig("nopreprocessing_Kuchen.png")
# plt.show()


#yvalues = [df["Text_processed_count"].sum(), df['Text_processed_gray_count'].sum(), df["Text_processed_otsu_count"].sum(), df["Text_processed_otsu_count_inverse"].sum(), df['Text_processed_niblack_count'].sum(), df["Text_processed_sauvola_count"].sum(), df["Text_processed_prima_count"].sum()]
#xvalues = ["no filter", "grayscale", "otsu", "otsu_inverse", "niblack", "sauvola", "prima"]
#yvalues = [df["Text_processed_count"].sum(), df['Text_processed_gray_count'].sum(), df["Text_processed_prima_count"].sum()]#,df["Text_processed_otsu_count"].sum(), df['Text_processed_niblack_count'].sum(), df["Text_processed_sauvola_count"].sum(), df["Text_processed_prima_count"].sum()]
yvalues = [df["Text_processed_count"].sum(), df["Text_processed_prima_count"].sum()]
xvalues = ["no filter", "prima"] #, "niblack", "sauvola", "otsu"]
plt.bar(xvalues, yvalues)
plt.xlabel("X-Werte")
plt.ylabel("Y-Werte")
plt.savefig("Thresholding_2.png")
plt.show()

print("Anzahl erkannter Wörter ohne Filter", df["Text_processed_count"].sum())
print("Anzahl erkannter Wörter Layoutparser", df["Text_processed_prima_count"].sum())