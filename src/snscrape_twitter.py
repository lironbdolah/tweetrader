import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import date, timedelta


# receives datetime.date objects
def date_range(start_date,end_date):
    delta = end_date - start_date   # returns timedelta
    dates = []
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        dates.append(day)
    return dates


def scrape_tweets(dates,keywords):
    tweets_list = []
    # list of dates
    dates1 = dates[:-1]
    dates2 = dates[1:]
    stop_count = 0 # in case the scraping fails
    tweet_count = 0
    tweet_daily_count = [] # another dataset that will count daily tweets


    # Using TwitterSearchScraper to scrape data and append tweets to list
    for k in keywords:
        for d1, d2 in zip(dates1, dates2):
                count = 0
                try:
                    for tweet in sntwitter.TwitterSearchScraper(k + ' since:' + str(d1) + ' until:' + str(d2)).get_items():
                            tweet_count += 1
                            #get tweet if the user has more then 10,000 followers
                            if tweet.user.followersCount > 5000:
                                tweets_list.append([d1, tweet.content,tweet.user.followersCount, tweet.user.verified])
                                count+=1
                            else:
                                continue
                            # limit the number of tweets per day
                            if count > 50:
                                print(d2)
                                tweet_daily_count.append([d1,tweet_count])
                                break
                    if count <= 50:
                        tweet_daily_count.append([d1, tweet_count])
                except:
                    # in case the scraper crashes during the process
                    tweets_df = pd.DataFrame(tweets_list, columns=['Date', 'Text','User_Followers','Verified'])
                    tweet_count_df = pd.DataFrame(tweet_daily_count , columns=['Date', 'tweet_count'])
                    words = k.split(' ')
                    tweets_df.to_csv('/nlp/' + words[0] + '_' + str(stop_count) +str(d1)+'.csv', index=False)
                    tweet_count_df.to_csv('/nlp/' + words[0] + '_' + str(
                        stop_count) +  str(d1) + '_tweet_count.csv', index=False)
                    print("file saved")
                    stop_count += 1

        # Creating a dataframe from the tweets list above
        tweets_df = pd.DataFrame(tweets_list,columns=['Date', 'Text','User_Followers','Verified'])
        tweet_count_df = pd.DataFrame(tweet_daily_count , columns=['Date', 'tweet_count'])
        words = k.split(' ')
        tweets_df.to_csv('/nlp/' + words[0] + '_cos_last.csv',index=False)
        tweet_count_df.to_csv('/nlp/' + words[0] + '_tweet_count.csv', index=False)
        print("file saved")




if __name__ == '__main__':
    dates = date_range(date(2017, 5, 1),date(2022, 5, 1))
    # define wanted key words
    keywords = [''] ### stocks you want to search for
    scrape_tweets(dates,keywords)
