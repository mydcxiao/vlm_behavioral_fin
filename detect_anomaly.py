import requests
import json
import time
import argparse
import os
import io
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_recency_bias(ticker, stock_file, eps_dir, window=5):
    """
    Analyze stock price movements around earnings report dates to detect potential recency bias in investor reactions. 
    The function examines a specified window of consecutive quarters, assessing if there is a consistent pattern in 
    how stock prices react to earnings surprises, either positive or negative.

    Parameters:
    - ticker (str): The stock ticker symbol for which the analysis is performed.
    - stock_file (pd.DataFrame): DataFrame containing historical stock prices with at least columns ['Date', 'Close'].
    - eps_dir (str): The directory path where JSON files containing earnings per share (EPS) data are stored.
    - window (int, optional): The number of consecutive quarters to consider for detecting the bias. Defaults to 4.

    Returns:
    - tuple of three lists:
        - bias_time (list of tuples): Each tuple contains the start and end date (as strings) of the period where bias was detected.
        - last (list of int): List of binary values where each corresponds to the direction of stock movement at the last observed earnings date within the window (1 if price went up or remained the same, 0 if down).
        - gt (list of int): List of binary values indicating the direction of stock movement at the most recent earnings date (1 if price went up or remained the same, 0 if down).

    Example:
    >>> detect_recency_bias('AAPL', stock_prices_df, '/path/to/eps/data', 4)
    ([('2020-01-01', '2020-04-01')], [1], [0])

    Notes:
    - This function assumes the EPS data is stored in JSON format and contains at least 'quarterlyEarnings' with keys 'reportedDate' and 'surprise'.
    - The stock data should be preloaded into the DataFrame with dates sorted in ascending order for accurate analysis.
    """
    dataframe = stock_file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    files = os.listdir(eps_dir)
    file = None
    for f in files:
        if ticker == f.split('.')[0].split('-')[1]:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(eps_dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    assert 'quarterlyEarnings' in eps_dict, f"'quarterlyEarnings' not found for {ticker} in {file}"
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df.sort_values(by='reportedDate', ascending=False)
    
    comp_stock.loc[:, 'Date'] = pd.to_datetime(comp_stock['Date'])
    quarterly_eps_df.loc[:, 'reportedDate'] = pd.to_datetime(quarterly_eps_df['reportedDate'])
    # EPS report within stock time range
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['reportedDate'].between(comp_stock['Date'].min(), comp_stock['Date'].max())]
    
    def _up_or_down(date):
        before = comp_stock.loc[comp_stock['Date'].between(date-pd.Timedelta(days=7), date), 'Close'].mean()
        after = comp_stock.loc[comp_stock['Date'].between(date+pd.Timedelta(days=1), date+pd.Timedelta(days=8)), 'Close'].mean()
        
        return int(after >= before)
    
    bias_time, last, gt = [], [], []
    for i in range(len(quarterly_eps_df) - window + 1):
        
        if quarterly_eps_df.iloc[i]['surprise'] == 'None':
            continue
        
        surprise = float(quarterly_eps_df.iloc[i]['surprise']) >= 0      
        last_up_or_down = None
        bias_diff = 0
        no_surprise_report = False
        for j in range(i+1, i+window):
            if quarterly_eps_df.iloc[j]['surprise'] == 'None':
                no_surprise_report = True
                break
            
            if (float(quarterly_eps_df.iloc[j]['surprise']) >= 0) == surprise:
                last_up_or_down = _up_or_down(quarterly_eps_df.iloc[j]['reportedDate']) if last_up_or_down is None else last_up_or_down
                if _up_or_down(quarterly_eps_df.iloc[j]['reportedDate']) != last_up_or_down:
                    bias_diff += 1
                else:
                    bias_diff -= 1
                    
        if bias_diff <= 0 or no_surprise_report:
            continue
        
        bias_time.append(((quarterly_eps_df.iloc[i+window-1]['reportedDate']-pd.Timedelta(days=30)).strftime('%Y-%m-%d'), quarterly_eps_df.iloc[i]['reportedDate'].strftime('%Y-%m-%d')))
        last.append(last_up_or_down)
        gt.append(_up_or_down(quarterly_eps_df.iloc[i]['reportedDate']))
    
    return bias_time, last, gt


def detect_primacy_bias():
    pass


def detect_authoritative_bias():
    pass


# if __name__ == '__main__':
#     bias_data, last, gt= detect_recency_bias("AAPL", pd.read_csv("data/stock_history.csv", header=[0, 1], index_col=0), "data/eps_history/")
#     print(len(bias_data))
#     print(bias_data)
#     print(len(last))
#     print(last)
#     print(len(gt))
#     print(gt)