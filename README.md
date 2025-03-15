<h1 align="center"> Stock Price Predicition with XGBoost </h1>
Millions of transactions are carried out every day in the Brazilian Financial Market, thousands of people buy and sell stocks every day betting on the price rising or falling. Its a very complex system, and the idea of this project is trying to predict the stock price of some shares in Brazilian Market. 

## Problem Definition 
The main idea is to try to use IA to predict the direction of the market in a few days, like in the interval of 1 or 3 days. 
Futhermore, I want to predict some specific pencetage change, i.e. if the stock moved like 1% or 2% for example.
For this problem I used the shares of PETR4 (Petrobras) and VALE3 (Vale). 

# Getting the Data
The first step was to get daily data for PETR4 and VALE3, I collected data using yfinance library. 
```python
ticker = "PETR4.SA"
dados = yf.download(ticker, period="15y", interval= "1d")
``` 





