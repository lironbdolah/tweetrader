import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import argrelextrema
import numpy as np
from matplotlib.pyplot import figure


class Broker():

    def __init__(self):
        self.cash = 10000
        self.stocks  = {}
        self.trade_history = []
        self.sharpes=[]

    def buy_stock(self,ticker,date,allocation,short):
        price=allocation*10000
        self.stocks[ticker] = [date,price,short]

    def sell_stock(self,ticker,sell_date,short,stock_data):
        buy_date = self.stocks[ticker][0]
        profit = get_profit(sell_date,buy_date,short,stock_data,ticker)
        print(ticker + ": "+ buy_date + "--" + sell_date + ", " + str(profit))
        self.cash+=profit
        self.trade_history.append([ticker,buy_date,sell_date,short,profit])
        self.stocks.pop(ticker)




def plot_trades(trade_history,ticks,stock_data):
    stock_data = stock_data.astype(float)
    figure(figsize=(16, 8), dpi=80)

    for tick in ticks:
        tick_data=stock_data[[tick]]
        actions = [i for i in trade_history if tick == i[0]]

        if len(actions)>0:

            tick_data['buy_long'] = None
            tick_data['sell_long'] = None
            tick_data['buy_short'] = None
            tick_data['sell_short'] = None

            for a in actions:
                if a[3]:
                    tick_data.loc[tick_data.index == a[1], 'buy_short'] = tick_data[tick].loc[a[1]]
                    tick_data.loc[tick_data.index == a[2], 'sell_short'] = tick_data[tick].loc[a[2]]
                else:
                    tick_data.loc[tick_data.index == a[1], 'buy_long'] = tick_data[tick].loc[a[1]]
                    tick_data.loc[tick_data.index == a[2], 'sell_long'] = tick_data[tick].loc[a[2]]

            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))

            plt.plot(tick_data.index, tick_data[tick])
            plt.scatter(tick_data.index, tick_data['buy_long'], c='g',zorder=2.5)
            plt.scatter(tick_data.index, tick_data['sell_long'], c='r',zorder=2.5)
            plt.scatter(tick_data.index, tick_data['buy_short'], c='purple', zorder=2.5)
            plt.scatter(tick_data.index, tick_data['sell_short'], c='orange', zorder=2.5)


            plt.title(tick + " trades")
            plt.savefig('/Users/lironbdolah/Documents/nlp/plots/' + tick + "_trades.png")
            plt.clf()

def train_test_split(df):
    train = df.iloc[:734,:]
    test =  df.iloc[734:,:]
    return train ,test



def loacl_edges(df):
    col = df.shape[1]
    figure(figsize=(16, 8), dpi=80)


    for i in ticks:
        n = 3  # number of points to be checked before and after
        df[i+'_max_tweets'] = df.iloc[argrelextrema(df[i+'_tweet_count'].values, np.greater_equal,
                                          order=n)[0]][i+'_tweet_count']


        df.loc[df[i+'_max_tweets'] < (df[i+'_tweet_count'].mean() + df[i+'_tweet_count'].std()) ,i+'_max_tweets'] = None

        # # plot top tweets
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        # plt.scatter(df.index, df[i+'_max_tweets'], c='g')
        # plt.plot(df.index, df[i+'_tweet_count'])
        # plt.title(i + " local max tweet count")
        # plt.savefig('/Users/lironbdolah/Documents/nlp/plots/' + i + "_tweet_count_max_min.png")
        # plt.clf()


        df[i + '_min_sentiment'] = df.iloc[argrelextrema(df[i + '_sentiment'].values, np.less_equal,
                                          order=n)[0]][i + '_sentiment']
        df[i + '_max_sentiment'] = df.iloc[argrelextrema(df[i + '_sentiment'].values, np.greater_equal,
                                               order=n)[0]][i + '_sentiment']

        # un-mark rows that dont match the criteria
        df.loc[df[i + '_max_sentiment'] < (df[i + '_sentiment'].mean()), i + '_max_sentiment'] = None
        df.loc[df[i + '_min_sentiment'] > (df[i + '_sentiment'].mean()), i + '_min_sentiment'] = None


        # # plot sentiment downfalls/highs
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        # plt.scatter(df.index, df[i + '_max_sentiment'], c='g')
        # plt.scatter(df.index, df[i + '_min_sentiment'], c='r')
        # plt.plot(df.index, df[i + '_sentiment'])
        # plt.title(i +" max/mins sentiments")
        # plt.savefig('/Users/lironbdolah/Documents/nlp/plots/' + i + "_sentiment_max_min.png")
        # plt.clf()


    df = df.iloc[:,col:]
    df = df.fillna(-1)
    return df



def execute_trades(df,ticks,broker,stock_data,allocation):
    for index,row in df.iterrows():
        for t in ticks:
            if index == '2017-07-03' and t == 'INTC':
                print()
            # execute long
            if row[t+'_max_sentiment'] >= 0 and row[t+'_max_tweets'] >= 0\
                    and t not in list(vars(broker)['stocks'].keys()):
                print("Buying: " + t  + ", (Long)")
                broker.buy_stock(t,index,allocation[t],False)

            # execute short
            if row[t+'_min_sentiment'] >= 0 and row[t+'_max_tweets'] >= 0\
                    and t not in list(vars(broker)['stocks'].keys()):
                print("Buying: " + t + ", (Short)")
                broker.buy_stock(t,index,allocation[t],True)

            # sell long
            if row[t + '_min_sentiment'] >= 0 and row[t+'_max_tweets'] >= 0\
                    and t in list(vars(broker)['stocks'].keys()) and not vars(broker)['stocks'][t][2]:
                print("Selling: " + t + ", (Long)")
                broker.sell_stock(t,index,False,stock_data)


            # sell short
            if row[t+'_max_sentiment'] >= 0 and row[t+'_max_tweets'] >= 0 \
                    and t in list(vars(broker)['stocks'].keys()) and vars(broker)['stocks'][t][2]:
                print("Selling: " + t + ", (Short)")
                broker.sell_stock(t, index, True,stock_data)






    #plot_trades(broker.trade_history,ticks,stock_data)

    ratio = shrape_ratio(broker.trade_history)

    print("The portfolio profit:")
    print(broker.cash)
    print()
    print("The portfolio Sharp Ratio:")
    print(ratio)
    print()
    print("stocks still in hand:")
    print(broker.stocks)



def shift_scores(df, ticks):

    scaler = MinMaxScaler()
    new_df = pd.DataFrame()


    for i in ticks:

        #shift scores:
        new_df[i+'_sentiment'] = df[i+'_sentiment'].shift(1)
        new_df[i+'_impact_score'] = df[i+'_impact_score'].shift(1)
        new_df[i+'_tweet_count'] = df[i+'_tweet_count'].shift(1)


    new_df = new_df.iloc[1:,:]

    #scale data
    new_df = pd.DataFrame(scaler.fit_transform(new_df), columns=new_df.columns)


    new_df['date'] = df['date'].tolist()[1:]
    new_df.set_index('date', inplace=True)
    return new_df


#calculate portfolio shrape ratio
def shrape_ratio(trade_history):

    number_of_transactions = len(trade_history)
    profits = [i[4] for i in trade_history]

    avg = sum(profits) / number_of_transactions
    deviations = math.sqrt(sum([(x - avg) ** 2 for x in profits]) / number_of_transactions)
    Rf = 0.0
    total_ex = 0
    for i in profits:
        ex = (1/number_of_transactions)*i
        total_ex+=ex

    sharpe = (total_ex-Rf)/deviations
    return sharpe

    # # calculate daily returns
    # tick_data['Daily_returns'] = tick_data[ticker].pct_change(1)
    #
    # Sharpe_Ratio = tick_data['Daily_returns'].mean() / tick_data['Daily_returns'].std()
    # return Sharpe_Ratio

#calculate the transaction profit
def get_profit(d2,d1,short,stock_data,ticker):

    # set date as index
    stock_data = stock_data.loc[d1:d2]
    stock_data[ticker] = pd.to_numeric(stock_data[ticker])
    x = stock_data.copy()
    stock_data[ticker + '_Norm_return'] = stock_data[ticker] / stock_data.iloc[0][ticker]
    stock_data[ticker + '_Allocation'] = stock_data[ticker + '_Norm_return']*allocation[ticker]
    stock_data[ticker + '_Position'] = stock_data[ticker + '_Allocation']*10000  # inital amount of money

    if short:
        profit = stock_data[ticker + '_Position'][d1] - stock_data[ticker + '_Position'][d2]
        print(profit)
    else:
        profit = stock_data[ticker + '_Position'][d2] - stock_data[ticker + '_Position'][d1]
        print(profit)

    return profit

# fit sentiment data dates to stock data dates
def remove_missing_dates(df,stock_data):
    new_header = stock_data.iloc[0]
    stock_data = stock_data[1:]
    stock_data.columns = new_header
    df['date'] = df.index
    dates = stock_data['date']
    df = df[df['date'].isin(dates)]

    return df,stock_data

if __name__ == '__main__':

    # load data
    df = pd.read_csv('complete.csv',index_col='date')
    stock_data = pd.read_csv('stock_data_n.csv')

    # default variables
    ticks = ['CSCO', 'BA', 'ESS', 'ORCL', 'INTC', 'GM', 'MSFT', 'EVRG', 'XOM', 'NFLX', 'CVS', 'JPM', 'PFE', 'AAPL',
             'BAX']
    stocks_pct = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05,
                  0.05]  ## % of each srock in the portfolio
    br = Broker()


    # remove days, where s&p500 is closed
    df,stock_data = remove_missing_dates(df,stock_data)
    stocks_name = stock_data.columns[:-1].tolist()
    allocation = {stocks_name[i]: stocks_pct[i] for i in range(len(stocks_name))}

    stock_data.set_index('date', inplace=True)

    # shift twitter data 1 day to fit the prior day closing prices
    df = shift_scores(df,ticks)

    # find local max/min for tweets and sentiments
    df = loacl_edges(df)

    in_sample,out_sample = train_test_split(df)

    execute_trades(in_sample,ticks,br,stock_data,allocation)
    execute_trades(out_sample, ticks, br, stock_data, allocation)
