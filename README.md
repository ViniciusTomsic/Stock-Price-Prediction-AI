<h1 align="center"> Stock Price Predicition with XGBoost </h1>
Millions of transactions are carried out every day in the Brazilian Financial Market, thousands of people buy and sell stocks every day betting on the price rising or falling. Its a very complex system, and the idea of this project is trying to predict the stock price of some shares in Brazilian Market. 

## Problem Definition 
The main idea is to try to use IA to predict the direction of the market in a few days, like in the interval of 1 or 3 days. 
Futhermore, I want to predict some specific pencetage change, i.e. if the stock moved like 1% or 2% for example.
For this problem I used the shares of PETR4 (Petrobras) and VALE3 (Vale). 

# Getting the Data
The first step was to get daily data for PETR4 and Ibovespa Index that will be used as reference in the data frame, I collected data using yfinance library.  

```python
ticker = "PETR4.SA"
data = yf.download(ticker, period="15y", interval= "1d")
data.to_csv("PETR4.csv")
``` 
In this example the period was set as fifteen years, but its good to take as many data as possible to train the model. 

Obs: yfinance can limit the number of requests that you can use so I needed to get that from MT5 when yfinance blocked me. 

# Processing data and creating Target
I created a funcition process_data to deal with the data frame, it takes 6 arguments: stock (the main stock), secondary_stock (used as reference), shift (an integer), up (True or False), percentage (the percentual change I want to detect) and test periods (an integer indicating the test set in days).
Next step was to create some new freatures for both stocks PETR4 and BVSP and adding the target to train the model. 
My data already had Closing Price, Open, High, Low and Volume, and to improve model performance I added SMA (slow moving average) with 21, 80, 200 and 1000 as periods. Besides that, I added STOCH, ATR and RSI to PETR4 stock. 

```python
df['STOCH'] = TA.STOCH(df)
df['ATR'] = TA.ATR(df)
df['RSI'] = TA.RSI(df)
df['SMA21'] = df['Close'] / TA.SMA(df, 21)
df['SMA80'] = df['Close'] / TA.SMA(df, 80)
df['SMA200'] = df['Close'] / TA.SMA(df, 200)
df['SMA1000'] = df['Close'] / TA.SMA(df, 1000)

df2['STOCH_2'] = TA.STOCH(df2)
df2['SMA21_2'] = df2['Close'] / TA.SMA(df2, 21)
df2['SMA80_2'] = df2['Close'] / TA.SMA(df2, 80)
```

The target takes 3 params in consideration shift, percentage and up (True or False), shift is the interval of days we are predicting, percentage is the variation we want to detect and up indicates if we are predicting the rise or fall of *Closing Price*. Note that shift is used as an interval of days and not as a step, i.e. we are trying to predict if the stock moved up or down by some percentage change in a window of 1 or more days, if the Closing Price is greater (or lower) in any of those days compared with today's Closing Price my target is 1, if not the target is set as 0. 


```python
if up:
    df['Target'] = (
        df['Close'].shift(-shift).rolling(window=shift, min_periods=1).max() > (1 + percentage) * df['Close']
    ).astype(int)
else:
    df['Target'] = (
        df['Close'].shift(-shift).rolling(window=shift, min_periods=1).min() < (1 - percentage) * df['Close']
    ).astype(int)
```
Lastly, the param test period is used to split the data, since the data is a time series we can't use the fucntion train_test_split or cross validation, the data must be in order. The process_data will return df and test if test period was specified, if not the function will return just the df  with both stocks information.

Note: This method using a window has proven to be better than predicting the market for a specific day in advance.

# Test model function
Before creating any model I've created a function to automate the training process, the function test_model takes 5 params: df, test, features, model and proba. 

1. df: is the train dataframe already split.
2. test: validation set to evaluate the model, the group of n days after train set. 
3. features: all the columns we want to use to train the model, excluding the target. 
4. model: its the model itself.
5. proba: is the threshold used to predict the target, default value is 0.5. 

This function is quite simple, it trains the model and uses predict proba to set a value for the prediction, if higher than the threshold the prediction is 1, if lower the predicted class is 0. Lastly, the function returns a df with the target and prediction for every day in the test set. 

# Models
With the automated process it was possible to test various IA models, I tried Ranfom Forest Classifier, XGBoost, LSTM, SVM and 1 dimensional Convolutional Network. 

## Metrics
To evaluate Supervised Learning models I used precision and recall, the main metric I wanted to optimize was precision, altought I noticed that some models had very high precisions, like 1. In those cases the recall was to low, so I changed the metric for optimization to F-beta score that balances recall and precision. I used beta equals 0.5 and 0.4 dependind on the case. 

Firstly, I initially thought that Neural Network methods would give best results, since LSTM is capable of memorize previous data and use this to adjust the weights, and Convulotional Neural Network is a strong method to preprocess data and finding patterns, though those. 






