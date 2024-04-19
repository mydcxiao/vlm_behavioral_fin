import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import requests
from bs4 import BeautifulSoup


def tickers_sp500():
    '''Downloads list of tickers currently listed in the S&P 500 '''
    #TODO check why does si.tickers_sp500() not work?
    # get list of all S&P 500 stocks 
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    sp_tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        if ticker == "BRK.B" or ticker == "BF.B":
            continue
        sp_tickers.append(ticker)
    
    return sp_tickers

# Function to fetch and save S&P 500 stock prices
def fetch_and_save_prices(tickers, start=None, end=None, save_path='stock_history.csv'):
    start = dt.datetime(2000, 1, 1).strftime('%Y-%m-%d') if not start else start # Start date
    end = dt.datetime.now().strftime('%Y-%m-%d') if not end else end # Format today's date
    prices = yf.download(tickers, start=start, end=end)
    # prices = prices['Close']  # We're interested in the closing prices
    prices.to_csv(save_path)  # Save to a CSV file
    return prices

# Main script
if __name__ == "__main__":
    tickers = tickers_sp500()
    tickers = tickers[:500]
    fetch_and_save_prices(tickers)