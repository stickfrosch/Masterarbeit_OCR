import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 10000)

merged_df = pd.read_pickle("Merged_Dataframe.pkl")
#print(merged_df)


merged_df['DailyExpress_real_words_count'] = merged_df['DailyExpress_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['DailyMail_real_words_count'] = merged_df['DailyMail_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['DailyMirror_real_words_count'] = merged_df['DailyMirror_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['DailyStar_real_words_count'] = merged_df['DailyStar_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['DailyTelegraph_real_words_count'] = merged_df['DailyTelegraph_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['FinancialTimes_real_words_count'] = merged_df['FinancialTimes_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['Guardian_real_words_count'] = merged_df['Guardian_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['i_real_words_count'] = merged_df['i_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['Metro_real_words_count'] = merged_df['Metro_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['TheTimes_real_words_count'] = merged_df['TheTimes_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['Sun_real_words_count'] = merged_df['Sun_real_words'].apply(lambda x: len([val for val in x.split()]))

yvalues = [merged_df['DailyExpress_real_words_count'].sum(), merged_df['DailyMail_real_words_count'].sum(), merged_df['DailyMirror_real_words_count'].sum(), merged_df['DailyStar_real_words_count'].sum(), merged_df['DailyTelegraph_real_words_count'].sum(), merged_df['FinancialTimes_real_words_count'].sum(), merged_df['Guardian_real_words_count'].sum(), merged_df['i_real_words_count'].sum(), merged_df['Metro_real_words_count'].sum(), merged_df['TheTimes_real_words_count'].sum(), merged_df['Sun_real_words_count'].sum()]
xvalues = ["Express", "Mail", "Mirror", "Star", "Telegraph", "Fin. Times", "Guardian", "i", "Metro", "Times", "Sun"]
plt.bar(xvalues, yvalues)
plt.xlabel("Zeitungen")
plt.ylabel("Anzahl erkannter Wörter")
plt.savefig("Number_real_words_Full.png")
plt.show()

#
merged_df['DailyExpress_real_words_>3'] = merged_df['DailyExpress_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['DailyMail_real_words_>3'] = merged_df['DailyMail_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['DailyMirror_real_words_>3'] = merged_df['DailyMirror_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['DailyStar_real_words_>3'] = merged_df['DailyStar_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['DailyTelegraph_real_words_>3'] = merged_df['DailyTelegraph_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['FinancialTimes_real_words_>3'] = merged_df['FinancialTimes_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['Guardian_real_words_>3'] = merged_df['Guardian_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['i_real_words_>3'] = merged_df['i_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['Metro_real_words_>3'] = merged_df['Metro_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['TheTimes_real_words_>3'] = merged_df['TheTimes_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
merged_df['Sun_real_words_>3'] = merged_df['Sun_real_words'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))


df_long_words = merged_df[["DailyExpress_real_words_>3", "DailyMail_real_words_>3", "DailyMirror_real_words_>3", "DailyStar_real_words_>3", "DailyTelegraph_real_words_>3", "FinancialTimes_real_words_>3", "Guardian_real_words_>3", "i_real_words_>3", "Metro_real_words_>3", "TheTimes_real_words_>3", "Sun_real_words_>3"]]

dailyexpress_list = merged_df['DailyExpress_real_words'].tolist()
dailymail_list = merged_df['DailyMail_real_words'].tolist()
dailymirror_list = merged_df['DailyMirror_real_words'].tolist()
dailystar_list = merged_df['DailyStar_real_words'].tolist()
dailytelegraph_list = merged_df['DailyTelegraph_real_words'].tolist()
financialtimes_list = merged_df['FinancialTimes_real_words'].tolist()
guardian_list = merged_df['Guardian_real_words'].tolist()
i_list = merged_df['i_real_words'].tolist()
metro_list = merged_df['Metro_real_words'].tolist()
thetimes_list = merged_df['TheTimes_real_words'].tolist()
sun_list = merged_df['Sun_real_words'].tolist()


def average_word_length(liste):
    string_1 = ""
    for word in liste:
        string_1 += word + " "
    numberofwords = len(string_1.split())
    lengthofwords = sum(list(map(len, string_1.split())))
    averagewordlength = lengthofwords / numberofwords
    return averagewordlength

print("Durchschnittliche Wortlänge Daily Express:", average_word_length(dailyexpress_list))
print("Durchschnittliche Wortlänge Daily Mail:", average_word_length(dailymail_list))
print("Durchschnittliche Wortlänge Daily Mirror:", average_word_length(dailymirror_list))
print("Durchschnittliche Wortlänge Daily Star:", average_word_length(dailystar_list))
print("Durchschnittliche Wortlänge Daily Telegraph:", average_word_length(dailytelegraph_list))
print("Durchschnittliche Wortlänge Financial Times:", average_word_length(financialtimes_list))
print("Durchschnittliche Wortlänge Guardian:", average_word_length(guardian_list))
print("Durchschnittliche Wortlänge i:", average_word_length(i_list))
print("Durchschnittliche Wortlänge Metro:", average_word_length(metro_list))
print("Durchschnittliche Wortlänge Times:", average_word_length(thetimes_list))
print("Durchschnittliche Wortlänge Sun:", average_word_length(sun_list))

yvalues = [average_word_length(dailyexpress_list), average_word_length(dailymail_list), average_word_length(dailymirror_list), average_word_length(dailystar_list), average_word_length(dailytelegraph_list), average_word_length(financialtimes_list),
           average_word_length(guardian_list), average_word_length(i_list), average_word_length(metro_list), average_word_length(thetimes_list), average_word_length(sun_list)]
xvalues = ["Express", "Mail", "Mirror", "Star", "Telegraph", "FT", "Guardian", "i", "Metro", "The Times", "Sun"]
plt.bar(xvalues, yvalues)
plt.xlabel("Zeitungen")
plt.ylabel("Durchschnittliche Länge erkannter Wörter")
plt.savefig("average_length_real_words.png")
plt.show()

merged_df['DailyExpress_Senti_Average'] = merged_df["DailyExpress_compound"].mean()
merged_df['DailyMail_Senti_Average'] = merged_df["DailyMail_compound"].mean()
merged_df['DailyMirror_Senti_Average'] = merged_df["DailyMirror_compound"].mean()
merged_df['DailyStar_Senti_Average'] = merged_df["DailyStar_compound"].mean()
merged_df['DailyTelegraph_Senti_Average'] = merged_df["DailyTelegraph_compound"].mean()
merged_df['FinancialTimes_Senti_Average'] = merged_df["FinancialTimes_compound"].mean()
merged_df['Guardian_Senti_Average'] = merged_df['Guardian_compound'].mean()
merged_df['i_Senti_Average'] = merged_df['i_compound'].mean()
merged_df['Metro_Senti_Average'] = merged_df['Metro_compound'].mean()
merged_df['TheTimes_Senti_Average'] = merged_df['TheTimes_compound'].mean()
merged_df['Sun_Senti_Average'] = merged_df['Sun_compound'].mean()

yvalues = [merged_df['DailyExpress_Senti_Average'].mean(), merged_df['DailyMail_Senti_Average'].mean(), merged_df['DailyMirror_Senti_Average'].mean(), merged_df['DailyStar_Senti_Average'].mean(), merged_df['DailyTelegraph_Senti_Average'].mean(), merged_df['FinancialTimes_Senti_Average'].mean(), merged_df['Guardian_Senti_Average'].mean(), merged_df['i_Senti_Average'].mean(), merged_df['Metro_Senti_Average'].mean(), merged_df['TheTimes_Senti_Average'].mean(), merged_df['Sun_Senti_Average'].mean()]
xvalues = ["Express", "Mail", "Mirror", "Star", "Telegraph", "Fin. Times", "Guardian", "i", "Metro", "Times", "Sun"]
plt.bar(xvalues, yvalues)
plt.xlabel("Zeitungen")
plt.ylabel("Durschnittliche Stimmung")
plt.savefig("Average_Sentiment.png")
plt.show()

averagesentiment_broadsheets = (merged_df['DailyTelegraph_Senti_Average'].mean() + merged_df['FinancialTimes_Senti_Average'].mean() + merged_df['Guardian_Senti_Average'].mean() + merged_df['i_Senti_Average'].mean() + merged_df['TheTimes_Senti_Average'].mean()) / 5
averagesentiment_tabloids = (merged_df['DailyExpress_Senti_Average'].mean() + merged_df['DailyMail_Senti_Average'].mean() + merged_df['DailyMirror_Senti_Average'].mean() + merged_df['DailyStar_Senti_Average'].mean() + merged_df['Sun_Senti_Average'].mean()) / 5
print("Durchschnittlicher Sentiment Broadsheets =", averagesentiment_broadsheets)
print("Durchschnittlicher Sentiment Tabloids =", averagesentiment_tabloids)

merged_df_sentiment = merged_df[['DailyExpress_Senti_Average', 'DailyMail_Senti_Average', 'DailyMirror_Senti_Average', 'DailyStar_Senti_Average', 'DailyTelegraph_Senti_Average', 'FinancialTimes_Senti_Average', 'Guardian_Senti_Average', 'i_Senti_Average', 'Metro_Senti_Average', 'TheTimes_Senti_Average', 'Sun_Senti_Average']]
sns.barplot(data=merged_df_sentiment)
plt.savefig("Average_Sentiment_corrected.png")
plt.show()

merged_df = merged_df.iloc[5:]

merged_df__Guardian_DStar = merged_df[["Date", "DailyStar_SMA_30", "Guardian_SMA_30"]]
print(merged_df__Guardian_DStar)
merged_df__Guardian_DStar.set_index("Date", inplace=True)
sns.lineplot(data=merged_df__Guardian_DStar)
plt.savefig("TimeSeries_Newspaper_Guardian_DStar.png")
plt.show()

AnzahlWörterOtsu = 2686
AnzahlWörteradaptiveotsu = 2424
AnzahlWörtersauvola = 2131
AnzahlWörterohne = 1094
AnzahlWörtergraustufe = 1002
AnzahlWörtersauvolaohne = 938

yvalues = [AnzahlWörterOtsu, AnzahlWörteradaptiveotsu, AnzahlWörtersauvola]
xvalues = ["Otsu", "Adaptive Otsu", "Sauvola"]
plt.bar(xvalues, yvalues)
plt.xlabel("Schwellwertmethoden")
plt.ylabel("Anzahl erkannter Wörter")
plt.savefig("Thresholdmethods.png")
plt.show()

yvalues = [AnzahlWörterOtsu, AnzahlWörterohne]
xvalues = ["Layoutparser", "ohne Layoutparser"]
plt.bar(xvalues, yvalues)
plt.xlabel("Seitensegmentierung")
plt.ylabel("Anzahl erkannter Wörter")
plt.savefig("Thresholdmethods_layoutparser.png")
plt.show()

DailyExpress = 4.987233955216222
DailyMail = 5.090712742980561
DailyMirror = 4.7864705033895865
DailyStar = 4.275650215031742
DailyTelegraph = 4.6473146927290765
FinancialTimes = 4.725635447500752
Guardian = 5.274432801475762
i = 5.509525614083439
Metro = 4.844324977940455
Times = 5.141311004784689
Sun = 4.540190102283294

yvalues = [DailyExpress, DailyMail, DailyMirror, DailyStar, Guardian, i, DailyTelegraph, Metro, Times, Sun, FinancialTimes]
xvalues = ["Daily Express", "Daily Mail", "Daily Mirror", "Daily Star", "Guardian", "i", "Daily Telegraph", "Metro", "Times", "Sun", "Financial Times"]
plt.bar(xvalues, yvalues)
plt.xlabel("Zeitungen")
plt.ylabel("Durchschnittliche Wortlänge")
plt.savefig("Meanwordlength.png")
plt.show()

averagewordlength_broadsheets = (DailyTelegraph + FinancialTimes + Guardian + i + Times) / 5
print("Durschnittliche Wortlänge Broadsheets", averagewordlength_broadsheets)
averagewordlength_tabloids = (DailyMail + DailyStar + DailyMirror + DailyExpress + Sun) / 5
print("Durschnittliche Wortlänge Tabloids", averagewordlength_tabloids)

DailyExpress = 0.6467391304347826
DailyMail = 0.5987318840579711
DailyMirror = 0.5914855072463768
DailyStar = 0.6512681159420289
DailyTelegraph = 0.5842391304347826
FinancialTimes = 0.605072463768116
Guardian = 0.572463768115942
i = 0.5661231884057971
Metro = 0.4990942028985507
TheTimes = 0.6213768115942029
Sun = 0.41032608695652173

yvalues = [DailyExpress, DailyMail, DailyMirror, DailyStar, Guardian, i, DailyTelegraph, Metro, TheTimes, Sun, FinancialTimes]
xvalues = ["Daily Express", "Daily Mail", "Daily Mirror", "Daily Star", "Guardian", "i", "Daily Telegraph", "Metro", "Times", "Sun", "Financial Times"]
plt.bar(xvalues, yvalues)
plt.xlabel("Zeitungen")
plt.ylabel("Lexikalische Diversität")
plt.savefig("Diversitylexical.png")
plt.show()

lexdiversity_broadsheets = (DailyTelegraph + FinancialTimes + Guardian + i + TheTimes) / 5
print("Durschnittliche lexikalische Diversität Broadsheets", lexdiversity_broadsheets)
lexdiversity_tabloids = (DailyMail + DailyStar + DailyMirror + DailyExpress + Sun) / 5
print("Durschnittliche lexikalische Diversität Tabloids", lexdiversity_tabloids)

merged_df = pd.read_pickle("Merged_Dataframe.pkl")
#print(merged_df)


merged_df['DailyExpress_real_words_count'] = merged_df['DailyExpress_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['DailyMail_real_words_count'] = merged_df['DailyMail_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['DailyMirror_real_words_count'] = merged_df['DailyMirror_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['DailyStar_real_words_count'] = merged_df['DailyStar_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['DailyTelegraph_real_words_count'] = merged_df['DailyTelegraph_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['FinancialTimes_real_words_count'] = merged_df['FinancialTimes_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['Guardian_real_words_count'] = merged_df['Guardian_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['i_real_words_count'] = merged_df['i_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['Metro_real_words_count'] = merged_df['Metro_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['TheTimes_real_words_count'] = merged_df['TheTimes_real_words'].apply(lambda x: len([val for val in x.split()]))
merged_df['Sun_real_words_count'] = merged_df['Sun_real_words'].apply(lambda x: len([val for val in x.split()]))

print("ExpressAnzahl =", merged_df['DailyExpress_real_words_count'].sum()/714)
print("DailyMail =", merged_df['DailyMail_real_words_count'].sum()/661)
print("DailyMirror =", merged_df['DailyMirror_real_words_count'].sum()/653)
print("DailyStar =", merged_df['DailyStar_real_words_count'].sum()/719)
print("DailyTelegraph =", merged_df['DailyTelegraph_real_words_count'].sum()/646)
print("FT =", merged_df['FinancialTimes_real_words_count'].sum()/668)
print("Guardian =", merged_df['Guardian_real_words_count'].sum()/632)
print("i =", merged_df['i_real_words_count'].sum()/625)
print("Metro =", merged_df['Metro_real_words_count'].sum()/551)
print("TheTimes =", merged_df['TheTimes_real_words_count'].sum()/687)
print("Sun =", merged_df['Sun_real_words_count'].sum()/453)

ExpressAnzahl = 68.23949579831933
DailyMail = 106.46898638426626
DailyMirror = 53.085758039816234
DailyStar = 33.95688456189151
DailyTelegraph = 341.64241486068113
FT = 293.8308383233533
Guardian = 166.40189873417722
i = 92.8864
Metro = 111.0671506352087
TheTimes = 304.2212518195051
Sun = 42.73289183222958

numberofwords_broadsheets = (DailyTelegraph + FT + Guardian + i + TheTimes) / 5
print("Durschnittliche Anzahl erkannter Wörter Broadsheets", numberofwords_broadsheets)
numberofwords_tabloids = (DailyMail + DailyStar + DailyMirror + ExpressAnzahl + Sun) / 5
print("Durschnittliche Anzahl erkannter Wörter Tabloids", numberofwords_tabloids)

yvalues = [ExpressAnzahl, DailyMail, DailyMirror, DailyStar, Guardian, i, DailyTelegraph, Metro, TheTimes, Sun, FT]
xvalues = ["Daily Express", "Daily Mail", "Daily Mirror", "Daily Star", "Guardian", "i", "Daily Telegraph", "Metro", "Times", "Sun", "Financial Times"]
plt.bar(xvalues, yvalues)
plt.xlabel("Zeitungen")
plt.ylabel("Durchschnittliche Anzahl erkannter Wörter")
plt.savefig("averagerealwords.png")
plt.show()

Express   =    0.026639
Mail       =   0.021749
Mirror      =  0.025999
Star         = 0.014039
Telegraph     = 0.013303
FinancialTimes  =  0.025221
Guardian    =  0.016888
i           =  0.015633
Metro       =  0.017790
Times       =  0.031784
Sun         =  0.009004

yvalues = [Express, Mail, Mirror, Star, Guardian, i, Telegraph, Metro, Times, Sun, FinancialTimes]
xvalues = ["Daily Express", "Daily Mail", "Daily Mirror", "Daily Star", "Guardian", "i", "Daily Telegraph", "Metro", "Times", "Sun", "Financial Times"]
plt.bar(xvalues, yvalues)
plt.xlabel("Zeitungen")
plt.ylabel("Varianz der Sentimentwerte")
plt.savefig("variancepernewspaper.png")
plt.show()

Express   =    0.163215
Mail       =   0.147475
Mirror     =   0.161241
Star        =  0.118488
Telegraph   =  0.115340
FinancialTimes  =  0.158811
Guardian   =   0.129954
i         =    0.125033
Metro    =     0.133378
Times     =    0.178281
Sun      =    0.094892

yvalues = [Express, Mail, Mirror, Star, Guardian, i, Telegraph, Metro, Times, Sun, FinancialTimes]
xvalues = ["Daily Express", "Daily Mail", "Daily Mirror", "Daily Star", "Guardian", "i", "Daily Telegraph", "Metro", "Times", "Sun", "Financial Times"]
plt.bar(xvalues, yvalues)
plt.xlabel("Zeitungen")
plt.ylabel("Standardabweichung der Sentimentwerte")
plt.savefig("standarddebrpernewspaper.png")
plt.show()

averagestdsentiment_broadsheets = (Telegraph + FinancialTimes + Guardian + i + Times) / 5
averagestdsentiment_tabloids = (Express + Mail + Mirror + Star + Sun) / 5
print("Durchschnittliche Standardabweichung der Sentimentwerte der Broadsheets = ", averagestdsentiment_broadsheets)
print("Durchschnittliche Standardabweichung der Sentimentwerte der Tabloids = ", averagestdsentiment_tabloids)

# wordcloud2 = WordCloud().generate(' '.join(merged_df['DailyExpress_real_words_>3']))
# plt.imshow(wordcloud2)
# plt.axis("off")
# plt.savefig("Wordcloud_DailyExpress")
# plt.show()
#
# wordcloud3 = WordCloud().generate(' '.join(merged_df['DailyMail_real_words_>3']))
# plt.imshow(wordcloud3)
# plt.axis("off")
# plt.savefig("Wordcloud_DailyMail")
# plt.show()
#
# wordcloud4 = WordCloud().generate(' '.join(merged_df['DailyMirror_real_words_>3']))
# plt.imshow(wordcloud4)
# plt.axis("off")
# plt.savefig("Wordcloud_DailyMirror")
# plt.show()
#
# wordcloud5 = WordCloud().generate(' '.join(merged_df['DailyStar_real_words_>3']))
# plt.imshow(wordcloud5)
# plt.axis("off")
# plt.savefig("Wordcloud_DailyStar")
# plt.show()
#
# wordcloud6 = WordCloud().generate(' '.join(merged_df['DailyTelegraph_real_words_>3']))
# plt.imshow(wordcloud6)
# plt.axis("off")
# plt.savefig("Wordcloud_DailyTelegraph")
# plt.show()
#
# wordcloud = WordCloud().generate(' '.join(merged_df['FinancialTimes_real_words_>3']))
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.savefig("Wordcloud_FinancialTimes")
# plt.show()
#
# wordcloud7 = WordCloud().generate(' '.join(merged_df['Guardian_real_words_>3']))
# plt.imshow(wordcloud7)
# plt.axis("off")
# plt.savefig("Wordcloud_Guardian")
# plt.show()
#
# wordcloud8 = WordCloud().generate(' '.join(merged_df['i_real_words_>3']))
# plt.imshow(wordcloud8)
# plt.axis("off")
# plt.savefig("Wordcloud_i")
# plt.show()
#
# wordcloud9 = WordCloud().generate(' '.join(merged_df['Metro_real_words_>3']))
# plt.imshow(wordcloud9)
# plt.axis("off")
# plt.savefig("Wordcloud_Metro")
# plt.show()
#
# wordcloud10 = WordCloud().generate(' '.join(merged_df['TheTimes_real_words_>3']))
# plt.imshow(wordcloud10)
# plt.axis("off")
# plt.savefig("Wordcloud_TheTimes")
# plt.show()
#
# wordcloud11 = WordCloud().generate(' '.join(merged_df['Sun_real_words_>3']))
# plt.imshow(wordcloud11)
# plt.axis("off")
# plt.savefig("Wordcloud_Sun")
# plt.show()



