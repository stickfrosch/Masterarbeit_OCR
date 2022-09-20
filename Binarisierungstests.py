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

    L = []
    L_gray = []
    L_otsu = []
    L_otsu_inverse = []
    L_niblack = []
    L_sauvola = []
    L_adaptive_mean = []
    L_adaptive_gaussian = []
    title = []

    #Iteriere über jedes Bild im Ordner
    for imageName in os.listdir(path):
        inputPath = os.path.join(path, imageName)
        if inputPath == path + "/.DS_Store":
            continue
        #img = Image.open(inputPath)

        image = cv2.imread(inputPath)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert2grayscale
        (thresh, binary) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # convert2binary
        (thresh, binary_inverse) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh_niblack = threshold_niblack(gray)
        img_niblack = gray > thresh_niblack
        thresh_sauvola = threshold_sauvola(gray)
        img_sauvola = gray > thresh_sauvola
        #img_adaptive_mean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                                  #cv2.THRESH_BINARY,11,2)
        #img_adaptive_gaussian = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)


        # OCR anwenden und in Liste L anfügen
        text = pt.image_to_string(image, lang="eng", config='--psm 11 -c thresholding_method=0')
        #text_gray = pt.image_to_string(gray, lang="eng")
        text_gray = pt.image_to_string(gray, lang="eng", config='--psm 11 -c thresholding_method=0')
        text_otsu = pt.image_to_string(image, lang="eng", config='--psm 11 -c thresholding_method=0')
        text_otsu_inverse = pt.image_to_string(gray, lang="eng", config='--psm 11 -c thresholding_method=0')
        text_niblack = pt.image_to_string(image, lang="eng", config='--psm 11 -c thresholding_method=0')
        text_sauvola = pt.image_to_string(img_sauvola, lang="eng", config='--psm 11 -c thresholding_method=0')
        #text_adaptive_mean = pt.image_to_string(img_adaptive_mean, lang="eng")
        #text_adaptive_gaussian = pt.image_to_string(img_adaptive_gaussian, lang="eng")

        L.append(text)
        L_gray.append(text_gray)
        L_otsu.append(text_otsu)
        L_otsu_inverse.append(text_otsu_inverse)
        #L_otsu_inverse.append(L_otsu)
        L_niblack.append(text_niblack)
        L_sauvola.append(text_sauvola)
        #L_adaptive_mean.append(text_adaptive_mean)
        #L_adaptive_gaussian.append(text_adaptive_gaussian)
        title.append(imageName)

        # Entferne '\n' für Absätze in L
        L = [w.replace('\n', '') for w in L]
        L_gray = [w.replace('\n', '') for w in L_gray]
        L_otsu = [w.replace('\n', '') for w in L_otsu]
        L_otsu_inverse = [w.replace('\n', '') for w in L_otsu_inverse]
        L_niblack = [w.replace('\n', '') for w in L_niblack]
        L_sauvola = [w.replace('\n', '') for w in L_sauvola]
        #L_adaptive_mean = [w.replace('\n', '') for w in L_adaptive_mean]
        #L_adaptive_gaussian = [w.replace('\n', '') for w in L_adaptive_gaussian]

    d = {"Name": title, "Text": L, "Text_gray": L_gray, "Text_otsu": L_otsu, "Text_otsu_inverse": L_otsu_inverse, "Text_niblack": L_niblack, "Text_sauvola": L_sauvola}
    #d = {"Name": title, "Text": L, "Text_gray": L_gray, "Text_otsu": L_otsu, "Text_otsu_inverse": L_otsu_inverse, "Text_niblack": L_niblack, "Text_sauvola": L_sauvola, "Text_adaptive_mean": L_adaptive_mean, "Text_adaptive_gaussian": L_adaptive_gaussian}
    df = pd.DataFrame(d)
    return df


if __name__ == '__main__':
    main()

df = main()

print(df)
def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def lemmatizing(sentence):
    tokens = word_tokenize(sentence)
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

df['Text_processed'] = df['Text'].astype(str).apply(lambda x: remove_punctuation(x))
df['Text_processed_gray'] = df['Text_gray'].astype(str).apply(lambda x: remove_punctuation(x))
df['Text_processed_otsu'] = df['Text_otsu'].astype(str).apply(lambda x: remove_punctuation(x))
df['Text_processed_otsu_inverse'] = df['Text_otsu_inverse'].astype(str).apply(lambda x: remove_punctuation(x))
df['Text_processed_niblack'] = df['Text_niblack'].astype(str).apply(lambda x: remove_punctuation(x))
df['Text_processed_sauvola'] = df['Text_sauvola'].astype(str).apply(lambda x: remove_punctuation(x))
#df['Text_processed_adaptive_mean'] = df['Text_adaptive_mean'].astype(str).apply(lambda x: remove_punctuation(x))
#df['Text_processed_adaptive_gaussian'] = df['Text_adaptive_gaussian'].astype(str).apply(lambda x: remove_punctuation(x))

df['Text_processed'] = df['Text_processed'].apply(lambda x: x.lower())
df['Text_processed_gray'] = df['Text_processed_gray'].apply(lambda x: x.lower())
df['Text_processed_otsu'] = df['Text_processed_otsu'].apply(lambda x: x.lower())
df['Text_processed_otsu_inverse'] = df['Text_processed_otsu_inverse'].apply(lambda x: x.lower())
df['Text_processed_niblack'] = df['Text_processed_niblack'].apply(lambda x: x.lower())
df['Text_processed_sauvola'] = df['Text_processed_sauvola'].apply(lambda x: x.lower())
#df['Text_processed_adaptive_mean'] = df['Text_processed_adaptive_mean'].apply(lambda x: x.lower())
#df['Text_processed_adaptive_gaussian'] = df['Text_processed_adaptive_gaussian'].apply(lambda x: x.lower())

df['Text_processed'] = df['Text_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Text_processed_gray'] = df['Text_processed_gray'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Text_processed_otsu'] = df['Text_processed_otsu'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Text_processed_otsu_inverse'] = df['Text_processed_otsu_inverse'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Text_processed_niblack'] = df['Text_processed_niblack'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Text_processed_sauvola'] = df['Text_processed_sauvola'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df['Text_processed_adaptive_mean'] = df['Text_processed_adaptive_mean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df['Text_processed_adaptive_gaussian'] = df['Text_processed_adaptive_gaussian'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df['Text_processed_token'] = df['Text_processed'].apply(word_tokenize)
df['Text_processed_gray_token'] = df['Text_processed_gray'].apply(word_tokenize)
df['Text_processed_otsu_token'] = df['Text_processed_otsu'].apply(word_tokenize)
df['Text_processed_otsu_token_inverse'] = df['Text_processed_otsu_inverse'].apply(word_tokenize)
df['Text_processed_niblack_token'] = df['Text_processed_niblack'].apply(word_tokenize)
df['Text_processed_sauvola_token'] = df['Text_processed_sauvola'].apply(word_tokenize)
#df['Text_processed_mean_token'] = df['Text_processed_adaptive_mean'].apply(word_tokenize)
#df['Text_processed_gaussian_token'] = df['Text_processed_adaptive_gaussian'].apply(word_tokenize)

df['Text_processed'] = df['Text_processed'].apply(lemmatizing)
df['Text_processed_gray'] = df['Text_processed_gray'].apply(lemmatizing)
df['Text_processed_otsu'] = df['Text_processed_otsu'].apply(lemmatizing)
df['Text_processed_otsu_inverse'] = df['Text_processed_otsu_inverse'].apply(lemmatizing)
df['Text_processed_niblack'] = df['Text_processed_niblack'].apply(lemmatizing)
df['Text_processed_sauvola'] = df['Text_processed_sauvola'].apply(lemmatizing)
#df['Text_processed_adaptive_mean'] = df['Text_processed_adaptive_mean'].apply(lemmatizing)
#df['Text_processed_adaptive_gaussian'] = df['Text_processed_adaptive_gaussian'].apply(lemmatizing)

df['Text_processed_count'] = df['Text_processed'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
df['Text_processed_gray_count'] = df['Text_processed_gray'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
#df['Text_processed_otsu_count'] = df['Text_processed_otsu'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
df['Text_processed_otsu_count'] = df['Text_processed_otsu'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
df['Text_processed_otsu_count_inverse'] = df['Text_processed_otsu_inverse'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
df['Text_processed_niblack_count'] = df['Text_processed_niblack'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
df['Text_processed_sauvola_count'] = df['Text_processed_sauvola'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
#df['Text_processed_mean_count'] = df['Text_processed_adaptive_mean'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))
#df['Text_processed_gaussian_count'] = df['Text_processed_adaptive_gaussian'].apply(lambda x: len([val for val in x.split() if val in english_words_1]))

#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text', 'Text_gray', 'Text_otsu', 'Text_niblack', 'Text_sauvola']].astype(str).apply(lambda x: lower(x))
#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']].apply(lambda x: x.lower())
#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']].apply(word_tokenize)
#df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']] = df[['Text_processed', 'Text_processed_gray', 'Text_processed_otsu', 'Text_processed_niblack', 'Text_processed_sauvola']].apply(lemmatizing)



sizes = df["Text_processed_count"]
labels = df["Name"]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig("nopreprocessing_Kuchen.png")
plt.show()


yvalues = [df["Text_processed_count"].sum(), df['Text_processed_gray_count'].sum(), df["Text_processed_otsu_count"].sum(), df["Text_processed_otsu_count_inverse"].sum(), df['Text_processed_niblack_count'].sum(), df["Text_processed_sauvola_count"].sum()]
xvalues = ["no filter", "grayscale", "otsu", "otsu_inverse", "niblack", "sauvola"]
plt.bar(xvalues, yvalues)
plt.xlabel("Schwellwert Methode")
plt.ylabel("Anzahl erkannter Wörter")
plt.savefig("Thresholding.png")
plt.show()

print(df)

print("Anzahl erkannter Wörter ohne Filter", df["Text_processed_count"].sum())
print("Anzahl erkannter Wörter graues Bild", df["Text_processed_gray_count"].sum())
print("Anzahl erkannter Wörter ohne Filter und Lokal Otsu", df["Text_processed_niblack_count"].sum())
print("Anzahl erkannter Wörter mit Grau und Lokal Otsu", df["Text_processed_otsu_count_inverse"].sum())
print("Anzahl erkannter Wörter ohne Filter und Sauvola", df["Text_processed_sauvola_count"].sum())
#df.plot.pie(y='count', figsize=(5, 5))
#plt.savefig("Kuchen.png")

# # Getting nlp from spacy.load
# nlp=spacy.load('en_core_web_sm')
# # Making the function to get the sentiments out of the dataframe
# def get_sentiment(data,name):
#     count=1
#     l=len(data)
#     positive_sentiments=[]
#     negative_sentiments=[]
#     for tex in data[name].values:
#         print('The current status is :',count*100/l,'%')
#         tex=nlp(tex)
#         noun=[]
#         verb=[]
#         adj=[]
#         adv=[]
#         for i in tex :
#             if i.pos_=='NOUN':
#                 noun.append(i)
#             elif i.pos_ =='ADJ':
#                 adj.append(i)
#             elif i.pos_ =='VERB':
#                 verb.append(i)
#             elif i.pos_=='ADV':
#                 adv.append(i)
#         clear_output(wait=True)
#         count+=1
#         neg_score=[]
#         pos_score=[]
#         for i in tex :
#             try:
#                 if i in noun:
#                     x=swn.senti_synset(str(i)+'.n.01')
#                     neg_score.append(x.neg_score())
#                     pos_score.append(x.pos_score())
#                 elif i in adj:
#                     x=swn.senti_synset(str(i)+'.a.02')
#                     neg_score.append(x.neg_score())
#                     pos_score.append(x.pos_score())
#                 elif i in adv :
#                     x=swn.senti_synset(str(i)+'.r.02')
#                     neg_score.append(x.neg_score())
#                     pos_score.append(x.pos_score())
#                 elif i in verb :
#                     x=swn.senti_synset(str(i)+'.v.02')
#                     neg_score.append(x.neg_score())
#                     pos_score.append(x.pos_score())
#
#             except:
#                 pass
#         positive_sentiments.append(np.mean(pos_score))
#         negative_sentiments.append(np.mean(neg_score))
#
#     df['Positive Sentiment']=positive_sentiments
#     df['Negative Sentiment']=negative_sentiments
#
# get_sentiment(df,'Text_processed')
#
# overall=[]
# for i in range(len(df)):
#     if df['Positive Sentiment'][i]>df['Negative Sentiment'][i]:
#         overall.append('Positive')
#     elif df['Positive Sentiment'][i]<df['Negative Sentiment'][i]:
#         overall.append('Negative')
#     else:
#         overall.append('Neutral')
# df['Overall Sentiment']=overall
# print(df)
#
# sns.countplot(df['Overall Sentiment'])
# plt.show()