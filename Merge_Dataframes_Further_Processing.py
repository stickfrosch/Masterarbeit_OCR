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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import plotly.express as px

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 10000)

def load_words():
    with open('english_words_lemmatized.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words

english_words_1 = load_words()

df = pd.read_pickle("DailyExpress.pkl")
df1 = pd.read_pickle("DailyMail.pkl")
df2 = pd.read_pickle("DailyMirror.pkl")
df3 = pd.read_pickle("DailyStar.pkl")
df4 = pd.read_pickle("DailyTelegraph.pkl")
df5 = pd.read_pickle("FinancialTimes.pkl")
df6 = pd.read_pickle("Guardian.pkl")
df7 = pd.read_pickle("i.pkl")
df8 = pd.read_pickle("Metro.pkl")
df9 = pd.read_pickle("TheTimes.pkl")
df10 = pd.read_pickle("Sun.pkl")

df = df.sort_values(by="datum")
df1 = df1.sort_values(by="datum")
df2 = df2.sort_values(by="datum")
df3 = df3.sort_values(by="datum")
df4 = df4.sort_values(by="datum")
df5 = df5.sort_values(by="datum")
df6 = df6.sort_values(by="datum")
df7 = df7.sort_values(by="datum")
df8 = df8.sort_values(by="datum")
df9 = df9.sort_values(by="datum")
df10 = df10.sort_values(by="datum")

df["Date"] = pd.to_datetime(df["datum"]).dt.date
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['DE_Time'] = pd.to_datetime(df['datum']).dt.time
df["DailyExpress_Text"] = df["Text_processed_prima"]
df = df[["DailyExpress_Text", "Date", "DE_Time"]]

df1["Date"] = pd.to_datetime(df1["datum"]).dt.date
df1['Date'] = pd.to_datetime(df1['Date'], format='%Y-%m-%d')
df1['DM_Time'] = pd.to_datetime(df1['datum']).dt.time
df1["DailyMail_Text"] = df1["Text_processed_prima"]
df1 = df1[["DailyMail_Text", "Date", "DM_Time"]]

df2["Date"] = pd.to_datetime(df2["datum"]).dt.date
df2['Date'] = pd.to_datetime(df2['Date'], format='%Y-%m-%d')
df2['DMirror_Time'] = pd.to_datetime(df2['datum']).dt.time
df2["DailyMirror_Text"] = df2["Text_processed_prima"]
df2 = df2[["DailyMirror_Text", "Date", "DMirror_Time"]]

df3["Date"] = pd.to_datetime(df3["datum"]).dt.date
df3['Date'] = pd.to_datetime(df3['Date'], format='%Y-%m-%d')
df3['DStar_Time'] = pd.to_datetime(df3['datum']).dt.time
df3["DailyStar_Text"] = df3["Text_processed_prima"]
df3 = df3[["DailyStar_Text", "Date", "DStar_Time"]]

df4["Date"] = pd.to_datetime(df4["datum"]).dt.date
df4['Date'] = pd.to_datetime(df4['Date'], format='%Y-%m-%d')
df4['DTelegraph_Time'] = pd.to_datetime(df4['datum']).dt.time
df4["DailyTelegraph_Text"] = df4["Text_processed_prima"]
df4 = df4[["DailyTelegraph_Text", "Date", "DTelegraph_Time"]]

df5["Date"] = pd.to_datetime(df5["datum"]).dt.date
df5['Date'] = pd.to_datetime(df5['Date'], format='%Y-%m-%d')
df5['FT_Time'] = pd.to_datetime(df5['datum']).dt.time
df5["FinancialTimes_Text"] = df5["Text_processed_prima"]
df5 = df5[["FinancialTimes_Text", "Date", "FT_Time"]]

df6["Date"] = pd.to_datetime(df6["datum"]).dt.date
df6['Date'] = pd.to_datetime(df6['Date'], format='%Y-%m-%d')
df6['Guardian_Time'] = pd.to_datetime(df6['datum']).dt.time
df6["Guardian_Text"] = df6["Text_processed_prima"]
df6 = df6[["Guardian_Text", "Date", "Guardian_Time"]]

df7["Date"] = pd.to_datetime(df7["datum"]).dt.date
df7['Date'] = pd.to_datetime(df7['Date'], format='%Y-%m-%d')
df7['i_Time'] = pd.to_datetime(df7['datum']).dt.time
df7["i_Text"] = df7["Text_processed_prima"]
df7 = df7[["i_Text", "Date", "i_Time"]]

df8["Date"] = pd.to_datetime(df8["datum"]).dt.date
df8['Date'] = pd.to_datetime(df8['Date'], format='%Y-%m-%d')
df8['Metro_Time'] = pd.to_datetime(df8['datum']).dt.time
df8["Metro_Text"] = df8["Text_processed_prima"]
df8 = df8[["Metro_Text", "Date", "Metro_Time"]]

df9["Date"] = pd.to_datetime(df9["datum"]).dt.date
df9['Date'] = pd.to_datetime(df9['Date'], format='%Y-%m-%d')
df9['Times_Time'] = pd.to_datetime(df9['datum']).dt.time
df9["TheTimes_Text"] = df9["Text_processed_prima"]
df9 = df9[["TheTimes_Text", "Date", "Times_Time"]]

df10["Date"] = pd.to_datetime(df10["datum"]).dt.date
df10['Date'] = pd.to_datetime(df10['Date'], format='%Y-%m-%d')
df10['Sun_Time'] = pd.to_datetime(df10['datum']).dt.time
df10["Sun_Text"] = df10["Text_processed_prima"]
df10 = df10[["Sun_Text", "Date", "Sun_Time"]]


rng = pd.date_range('2019-04-11', '2022-04-09', freq='D')
df_date = pd.DataFrame({'Date': rng})
#print(df_date)

merged_df = pd.merge(df_date, df, how="outer", on="Date")
merged_df = pd.merge(merged_df, df1, how="outer", on="Date")
merged_df = pd.merge(merged_df, df2, how="outer", on="Date")
merged_df = pd.merge(merged_df, df3, how="outer", on="Date")
merged_df = pd.merge(merged_df, df4, how="outer", on="Date")
merged_df = pd.merge(merged_df, df5, how="outer", on="Date")
merged_df = pd.merge(merged_df, df6, how="outer", on="Date")
merged_df = pd.merge(merged_df, df7, how="outer", on="Date")
merged_df = pd.merge(merged_df, df8, how="outer", on="Date")
merged_df = pd.merge(merged_df, df9, how="outer", on="Date")
merged_df = pd.merge(merged_df, df10, how="outer", on="Date")

merged_df['DailyExpress_Text'] = merged_df['DailyExpress_Text'].fillna("")
merged_df['DailyMail_Text'] = merged_df['DailyMail_Text'].fillna("")
merged_df['DailyMirror_Text'] = merged_df['DailyMirror_Text'].fillna("")
merged_df['DailyStar_Text'] = merged_df['DailyStar_Text'].fillna("")
merged_df['DailyTelegraph_Text'] = merged_df['DailyTelegraph_Text'].fillna("")
merged_df['FinancialTimes_Text'] = merged_df['FinancialTimes_Text'].fillna("")
merged_df['Guardian_Text'] = merged_df['Guardian_Text'].fillna("")
merged_df['i_Text'] = merged_df['i_Text'].fillna("")
merged_df['Metro_Text'] = merged_df['Metro_Text'].fillna("")
merged_df['TheTimes_Text'] = merged_df['TheTimes_Text'].fillna("")
merged_df['Sun_Text'] = merged_df['Sun_Text'].fillna("")

#print(merged_df)
#print(merged_df.dtypes)

merged_df['DailyExpress_real_words'] = merged_df['DailyExpress_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['DailyMail_real_words'] = merged_df['DailyMail_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['DailyMirror_real_words'] = merged_df['DailyMirror_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['DailyStar_real_words'] = merged_df['DailyStar_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['DailyTelegraph_real_words'] = merged_df['DailyTelegraph_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['FinancialTimes_real_words'] = merged_df['FinancialTimes_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['Guardian_real_words'] = merged_df['Guardian_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['i_real_words'] = merged_df['i_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['Metro_real_words'] = merged_df['Metro_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['TheTimes_real_words'] = merged_df['TheTimes_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))
merged_df['Sun_real_words'] = merged_df['Sun_Text'].apply(lambda x: ' '.join([word for word in x.split() if word in english_words_1]))


sid = SentimentIntensityAnalyzer()

merged_df['DailyExpress_scores'] = merged_df['DailyExpress_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['DailyMail_scores'] = merged_df['DailyMail_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['DailyMirror_scores'] = merged_df['DailyMirror_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['DailyStar_scores'] = merged_df['DailyStar_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['DailyTelegraph_scores'] = merged_df['DailyTelegraph_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['FinancialTimes_scores'] = merged_df['FinancialTimes_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['Guardian_scores'] = merged_df['Guardian_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['i_scores'] = merged_df['i_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['Metro_scores'] = merged_df['Metro_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['TheTimes_scores'] = merged_df['TheTimes_real_words'].apply(lambda review: sid.polarity_scores(review))
merged_df['Sun_scores'] = merged_df['Sun_real_words'].apply(lambda review: sid.polarity_scores(review))

merged_df['DailyExpress_compound'] = merged_df['DailyExpress_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['DailyMail_compound'] = merged_df['DailyMail_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['DailyMirror_compound'] = merged_df['DailyMirror_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['DailyStar_compound'] = merged_df['DailyStar_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['DailyTelegraph_compound'] = merged_df['DailyTelegraph_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['FinancialTimes_compound'] = merged_df['FinancialTimes_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['Guardian_compound'] = merged_df['Guardian_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['i_compound'] = merged_df['i_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['Metro_compound'] = merged_df['Metro_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['TheTimes_compound'] = merged_df['TheTimes_scores'].apply(lambda score_dict: score_dict['compound'])
merged_df['Sun_compound'] = merged_df['Sun_scores'].apply(lambda score_dict: score_dict['compound'])
#df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')
#print(df['compound'].describe())

compound_df = merged_df[["DailyExpress_compound", "DailyMail_compound", "DailyMirror_compound", "DailyStar_compound", "DailyTelegraph_compound", "FinancialTimes_compound", "Guardian_compound", "i_compound", "Metro_compound", "TheTimes_compound", "Sun_compound"]]
print(compound_df.describe())

def lexical_diversity(text):
    return len(set(text)) / len(text)

print("Diversität der Wörter des Daily Express", lexical_diversity(merged_df['DailyExpress_real_words']))
print("Diversität der Wörter des Daily Mail", lexical_diversity(merged_df['DailyMail_real_words']))
print("Diversität der Wörter des Daily Mirror", lexical_diversity(merged_df['DailyMirror_real_words']))
print("Diversität der Wörter des Daily Star", lexical_diversity(merged_df['DailyStar_real_words']))
print("Diversität der Wörter des Daily Telegraph", lexical_diversity(merged_df['DailyTelegraph_real_words']))
print("Diversität der Wörter der Financial Times", lexical_diversity(merged_df['FinancialTimes_real_words']))
print("Diversität der Wörter des Guardian", lexical_diversity(merged_df['Guardian_real_words']))
print("Diversität der Wörter der i", lexical_diversity(merged_df['i_real_words']))
print("Diversität der Wörter der Metro", lexical_diversity(merged_df['Metro_real_words']))
print("Diversität der Wörter der The Times", lexical_diversity(merged_df['TheTimes_real_words']))
print("Diversität der Wörter der Sun", lexical_diversity(merged_df['Sun_real_words']))

# the simple moving average over a period of 30 days
merged_df['DailyExpress_SMA_30'] = merged_df["DailyExpress_compound"].rolling(30, min_periods=1).mean()
merged_df['DailyMail_SMA_30'] = merged_df["DailyMail_compound"].rolling(30, min_periods=1).mean()
merged_df['DailyMirror_SMA_30'] = merged_df["DailyMirror_compound"].rolling(30, min_periods=1).mean()
merged_df['DailyStar_SMA_30'] = merged_df["DailyStar_compound"].rolling(30, min_periods=1).mean()
merged_df['DailyTelegraph_SMA_30'] = merged_df["DailyTelegraph_compound"].rolling(30, min_periods=1).mean()
merged_df['FinancialTimes_SMA_30'] = merged_df["FinancialTimes_compound"].rolling(30, min_periods=1).mean()
merged_df['Guardian_SMA_30'] = merged_df['Guardian_compound'].rolling(30, min_periods=1).mean()
merged_df['i_SMA_30'] = merged_df['i_compound'].rolling(30, min_periods=1).mean()
merged_df['Metro_SMA_30'] = merged_df['Metro_compound'].rolling(30, min_periods=1).mean()
merged_df['TheTimes_SMA_30'] = merged_df['TheTimes_compound'].rolling(30, min_periods=1).mean()
merged_df['Sun_SMA_30'] = merged_df['Sun_compound'].rolling(30, min_periods=1).mean()

merged_df = merged_df.iloc[5:]

merged_df["Broadsheets"] = merged_df[["DailyTelegraph_SMA_30", "FinancialTimes_SMA_30", "Guardian_SMA_30", "i_SMA_30", "TheTimes_SMA_30"]].mean(axis=1)
merged_df["Tabloids"] = merged_df[["DailyExpress_SMA_30", "DailyMail_SMA_30", "DailyMirror_SMA_30", "DailyStar_SMA_30", "Sun_SMA_30"]].mean(axis=1)

merged_df__broadsheets_tabloids = merged_df[["Date", "Broadsheets", "Tabloids"]]
merged_df__broadsheets_tabloids.set_index("Date", inplace=True)
sns.lineplot(data=merged_df__broadsheets_tabloids)
plt.savefig("TimeSeries_Newspaper_Broadsheets_Tabloids.png")
plt.show()

merged_df_30 = merged_df[["Date", "DailyExpress_SMA_30", "DailyMail_SMA_30", "DailyMirror_SMA_30", "DailyStar_SMA_30", "DailyTelegraph_SMA_30", "FinancialTimes_SMA_30", "Guardian_SMA_30", "i_SMA_30", "Metro_SMA_30", "TheTimes_SMA_30", "Sun_SMA_30"]]
merged_df_30 = merged_df_30.rename(columns={"DailyExpress_SMA_30": "Daily Express", "DailyMail_SMA_30": "Daily Mail", "DailyMirror_SMA_30": "Daily Mirror", "DailyStar_SMA_30": "Daily Star", "DailyTelegraph_SMA_30": "Daily Telegraph", "FinancialTimes_SMA_30": "Financial Times", "Guardian_SMA_30": "Guardian", "i_SMA_30": "i", "Metro_SMA_30": "Metro", "TheTimes_SMA_30": "Times", "Sun_SMA_30": "Sun"})
merged_df_30.set_index("Date", inplace=True)
sns.lineplot(data=merged_df_30)
plt.savefig("TimeSeries_Newspaper_30.png")
plt.show()

#merged_df__Guardian_DStar = merged_df[["Date", "DailyStar_SMA_30", "Guardian_SMA_30"]]
merged_df__Guardian_DStar = merged_df_30[["Daily Star", "Guardian"]]
#print(merged_df__Guardian_DStar)
#merged_df__Guardian_DStar.set_index("Date", inplace=True)
sns.lineplot(data=merged_df__Guardian_DStar)
plt.savefig("TimeSeries_Newspaper_Guardian_DStar.png")
plt.show()

#merged_df__Telegraph_Times = merged_df[["Date", "DailyTelegraph_SMA_30", "TheTimes_SMA_30"]]
merged_df__Telegraph_Times = merged_df_30[["Daily Telegraph", "Times"]]
merged_df__Telegraph_Times = merged_df__Telegraph_Times.rename(columns={"Daily Telegraph": "DailyTelegraph"})
#merged_df__Telegraph_Times.set_index("Date", inplace=True)
sns.lineplot(data=merged_df__Telegraph_Times)
plt.savefig("TimeSeries_Newspaper_Telegraph_Times40.png")
plt.show()
print(merged_df__Telegraph_Times[merged_df__Telegraph_Times.Times == merged_df__Telegraph_Times.Times.max()])
print(merged_df__Telegraph_Times[merged_df__Telegraph_Times.DailyTelegraph == merged_df__Telegraph_Times.DailyTelegraph.max()])

#merged_df_Sun_DailyExpress = merged_df[["Date", "DailyExpress_SMA_30", "Sun_SMA_30"]]
merged_df_Sun_DailyExpress = merged_df_30[["Daily Express", "Sun"]]
sns.lineplot(data=merged_df_Sun_DailyExpress)
plt.savefig("Sun_DExpress_Sentiment.png")
plt.show()

#print(merged_df_30.corr())
fig = px.imshow(merged_df_30.corr())
fig.show()

merged_df_90 = merged_df[["Date", "DailyExpress_SMA_30", "DailyMail_SMA_30", "DailyMirror_SMA_30", "DailyStar_SMA_30", "DailyTelegraph_SMA_30", "FinancialTimes_SMA_30", "Guardian_SMA_30", "i_SMA_30", "Metro_SMA_30", "TheTimes_SMA_30", "Sun_SMA_30"]]
merged_df_90 = merged_df_90.rename(columns={"DailyExpress_SMA_30": "Express", "DailyMail_SMA_30": "Mail", "DailyMirror_SMA_30": "Mirror", "DailyStar_SMA_30": "Star", "DailyTelegraph_SMA_30": "Telegraph", "FinancialTimes_SMA_30": "Fin. Times", "Guardian_SMA_30": "Guardian", "i_SMA_30": "i", "Metro_SMA_30": "Metro", "TheTimes_SMA_30": "Times", "Sun_SMA_30": "Sun"})
merged_df_90.set_index("Date", inplace=True)
print("Varianzen:")
print(merged_df_90.var())

print("Standardabweichung:")
print(merged_df_90.std())

korrelationsdf = merged_df_90.corr()
plt.figure(figsize = (40,19))
plt.title('Korrelationsmatrix der Sentimentwerte')

sns.heatmap(korrelationsdf, annot=True, cmap='RdYlBu_r', fmt= '.4g',)
plt.show()
# merged_df.to_pickle("Merged_Dataframe.pkl")

# # the simple moving average over a period of 10 days
# merged_df['DailyExpress_SMA_10'] = merged_df["DailyExpress_compound"].rolling(10, min_periods=1).mean()
# merged_df['DailyMail_SMA_10'] = merged_df["DailyMail_compound"].rolling(10, min_periods=1).mean()
# merged_df['DailyMirror_SMA_10'] = merged_df["DailyMirror_compound"].rolling(10, min_periods=1).mean()
# merged_df['DailyStar_SMA_10'] = merged_df["DailyStar_compound"].rolling(10, min_periods=1).mean()
# merged_df['DailyTelegraph_SMA_10'] = merged_df["DailyTelegraph_compound"].rolling(10, min_periods=1).mean()
# merged_df['FinancialTimes_SMA_10'] = merged_df["FinancialTimes_compound"].rolling(10, min_periods=1).mean()
#
# merged_df_10 = merged_df[["Date", "DailyExpress_SMA_10", "DailyMail_SMA_10", "DailyMirror_SMA_10", "DailyStar_SMA_10", "DailyTelegraph_SMA_10", "FinancialTimes_SMA_10"]]
# sns.lineplot(data=merged_df_10)
# plt.savefig("TimeSeries_Newspaper_10.png")
# plt.show()

# wordcloud2 = WordCloud().generate(' '.join(merged_df['DailyExpress_real_words']))
# plt.imshow(wordcloud2)
# plt.axis("off")
# plt.savefig("Wordcloud_DailyExpress")
# plt.show()
#
