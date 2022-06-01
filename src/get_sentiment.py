import pandas as pd
import re
import flair
from deep_translator import GoogleTranslator
from langdetect import detect
import numpy as np

# cleans 1 string at a time
def clean_text(text,keywords):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    for k in keywords:
        word = '@' + k
        t = re.compile(r"(?i)" + word + "(?=\b)")
        text = t.sub(k, text)

    text = whitespace.sub(' ', text)
    text = web_address.sub('', text)
    text = user.sub('', text)
    text = text.replace('RT :', '') # remove the retweet
    text = emoji_pattern.sub(r'', text) # remove emoji  from text
    text = re.sub("\d+", "", text) # remove integers

    return text

# gets each tweet sentimental score (between 1 to -1)
def get_sentiment(tweet):
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    sentence = flair.data.Sentence(tweet)
    sentiment_model.predict(sentence)
    try:
        sentiment = sentence.labels[0].score  # numerical value 0-1
        if sentence.labels[0].value == 'NEGATIVE':
            sentiment += -1
    except:
        sentiment = 0.0
        return sentiment

    return sentiment

#calculate an avg daily sentiment score
def daily_sentiment(df,tweets_per_day):

    new_df = pd.DataFrame()
    df['date'] = pd.to_datetime(df['Datetime']).dt.date
    unique_dates = df['date'].unique()

    new_df['daily_sentiment'] = df['sentiment']
    new_df['impact_score'] = df['User_Followers']

    new_df = new_df.groupby(np.arange(len(new_df)) // tweets_per_day * len(unique_dates)).mean()
    new_df['date'] = unique_dates

    from sklearn.preprocessing import StandardScaler
    # normalize impact score to 0-1
    new_df['impact_score'] = StandardScaler().fit_transform(new_df['impact_score'].to_numpy().reshape(-1, 1))

    return new_df

#calculate each tweet impact score
def impact_score(df):
    df["User_Followers"] = pd.to_numeric(df["User_Followers"], downcast="float")
    df.User_Followers = df.apply(lambda row: row.User_Followers * 1.2 if row.Verified else row.User_Followers, axis=1)
    return df


def sort_data(keywords,stock_text_data):
    cleaned_text,sentiments = [],[]


    for index, row in stock_text_data.iterrows():
        text = clean_text(row['Text'], keywords)
        # detect language
        try:
            if detect(text) != 'en':
                text = GoogleTranslator(source='auto', target='en').translate(text) # translate text to English
                text = clean_text(text, keywords)  # clean raw text
                sentiment = get_sentiment(text)  # get sentiment for the clean text
            else:
                sentiment = get_sentiment(text)


        except: # probably spam
            print(row['Text'])
            sentiment = 0.0
            text='XXX'


        cleaned_text.append(text)
        sentiments.append(sentiment)


    stock_text_data = stock_text_data.drop('Text', 1)

    try:
        # add data to df
        stock_text_data['cleaned_text'] = cleaned_text
        stock_text_data['sentiment'] = sentiments

        # save temp clean text
        stock_text_data.to_csv('C:/Users/bdola/OneDrive/Documents/Order/At/nlp/2_cleaned_text/gme.csv',index=False)

        # calculate daily sentiment:
        new_df = daily_sentiment(stock_text_data,100)
        new_df.to_csv('C:/Users/bdola/OneDrive/Documents/Order/At/nlp/3_daily_sentiment/gme.csv', index=False)

    except:
        new_df = pd.DataFrame()
        # add data to df
        new_df['cleaned_text'] = cleaned_text
        new_df['sentiment'] = sentiments

        # save temp data
        new_df.to_csv('C:/Users/bdola/OneDrive/Documents/Order/At/nlp/2_cleaned_text/gme.csv', index=False)

        # calculate daily sentiment:
        new_df = daily_sentiment(new_df)
        new_df.to_csv('C:/Users/bdola/OneDrive/Documents/Order/At/nlp/3_daily_sentiment/gme.csv', index=False)





if __name__ == '__main__':

    stock_text_data = pd.read_csv('C:/Users/bdola/OneDrive/Documents/Order/At/nlp/1_raw_data/GME.csv')
    #stock_text_data =stock_text_data.iloc[0:2001,:]
    keywords = ['GOOG','Alphabet']
    stock_text_data = impact_score(stock_text_data)
    sort_data(keywords,stock_text_data)
