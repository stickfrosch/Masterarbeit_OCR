import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import numpy as np


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