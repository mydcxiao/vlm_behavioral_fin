import requests
import json
import time
import argparse
import sys
import os
import io
import re
import random
import warnings
import base64
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
from PIL import Image
from tqdm import tqdm
from functools import partial

from detect_anomaly import detect_recency_bias, detect_authoritative_bias
from data.fetch_sp500 import tickers_sp500, fetch_and_save_prices, fetch_and_save_eps


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps_key', type=str, default="", help='API key for fetching EPS data')
    parser.add_argument('--collect_data', action='store_true', default=False, help='Collect data or not')
    parser.add_argument('--collect_price', action='store_true', default=False, help='Collect stock price data')
    parser.add_argument('--collect_eps', action='store_true', default=False, help='Collect EPS data')
    parser.add_argument('--output_dir', type=str, default="images", help='Output directory')
    parser.add_argument('--stock_file', type=str, default="data/stock_history.csv", help='history stock data file path')
    parser.add_argument('--eps_dir', type=str, default="data/eps_history/", help='history eps data dir path')
    parser.add_argument('--ticker', type=str, nargs='*', help='list of stock ticker')
    parser.add_argument('--bias_type', type=str, default="recency", help='bias type')
    parser.add_argument('--window_size', type=int, default=5, help='window size for bias detection')
    parser.add_argument('--num_samples', type=int, default=100, help='number of samples to test')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    return parser.parse_args()


def load_stock_data(path):
    prices = pd.read_csv(path, header=[0, 1], index_col=0)
    
    return prices


def construct_images(file, dir, ticker, start, end):
    dataframe = file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock = comp_stock.loc[start:end]
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    files = os.listdir(dir)
    file = None
    for f in files:
        if ticker == f.split('.')[0].split('-')[1]:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['reportedDate'].between(start, end)]

    comp_stock.loc[:, 'Date'] = pd.to_datetime(comp_stock['Date'])
    comp_stock = comp_stock.infer_objects() # suppress warning
    comp_stock.set_index('Date', inplace=True)
    # Ensure quarterly_eps_df['reportedDate'] and quarterly_eps_df['fiscalDateEnding'] are in datetime format
    quarterly_eps_df['reportedDate'] = pd.to_datetime(quarterly_eps_df['reportedDate'])
    quarterly_eps_df['fiscalDateEnding'] = pd.to_datetime(quarterly_eps_df['fiscalDateEnding'])
    # Creating markers for EPS report date and fiscal end date
    quarterly_eps_df['surprise'] = pd.to_numeric(quarterly_eps_df['surprise'])
    eps_data = quarterly_eps_df.loc[quarterly_eps_df['reportedDate'].between(start, end), ['reportedDate', 'surprise']]
    meet_eps_data = eps_data.loc[eps_data['surprise'] >= 0]
    meet_eps_markers = meet_eps_data['reportedDate'].values
    meet_eps_surprise = meet_eps_data['surprise'].values
    miss_eps_data = eps_data.loc[eps_data['surprise'] < 0]
    miss_eps_markers = miss_eps_data['reportedDate'].values
    miss_eps_surprise = miss_eps_data['surprise'].values
    fiscal_markers = quarterly_eps_df.loc[quarterly_eps_df['fiscalDateEnding'].between(start, end), 'fiscalDateEnding'].values
    # Calculate minimum and maximum prices for y-axis scaling
    price_min = comp_stock[['Low']].min().min()  # min of 'Low' prices
    price_max = comp_stock[['High']].max().max()  # max of 'High' prices
    price_range = price_max - price_min
    price_buffer = price_range * 0.1  # 10% price buffer on each side
    # Calculate adaptive offsets for markers
    offset = price_range * 0.15
    # Creating EPS signals
    all_dates = pd.date_range(start, end, freq='D')
    low_prices = comp_stock['Low'].reindex(all_dates).interpolate(method='linear')
    high_prices = comp_stock['High'].reindex(all_dates).interpolate(method='linear')
    meet_eps_signal = high_prices.loc[meet_eps_markers] + offset
    miss_eps_signal = high_prices.loc[miss_eps_markers] + offset
    fiscal_signal = low_prices.loc[fiscal_markers] - offset
    # Setting figure size dynamically based on the date range
    date_range = (dt.datetime.strptime(end, '%Y-%m-%d') - dt.datetime.strptime(start, '%Y-%m-%d')).days
    fig_width = max(10, min(date_range / 30, 30)) # 30 days per inch
    fig_height = 6
    # Plotting the candlestick chart
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            fig, axlist = mpf.plot(
                comp_stock, type='candle', mav=(7), style='yahoo', 
                panel_ratios=(2,1), 
                figratio=(fig_width, fig_height),
                figscale=1 * min(1, date_range / 120),
                title=f'{ticker} stock price chart with EPS Dates', ylabel='Stock Price', 
                volume=True, show_nontrading=True, returnfig=True,
                ylim=(price_min - price_buffer, price_max + price_buffer),
            )
        except:
            fig, axlist = mpf.plot(
                comp_stock, type='line', mav=(), style='yahoo', 
                panel_ratios=(2,1), 
                figratio=(fig_width, fig_height),
                figscale=1 * min(1, date_range / 120),
                title=f'{ticker} stock price chart with EPS Dates', ylabel='Stock Price', 
                volume=True, show_nontrading=True, returnfig=True,
                ylim=(price_min - price_buffer, price_max + price_buffer),
            )       

    meet_eps_x = [mdates.date2num(date) for date in meet_eps_signal.index]
    meet_eps_y = meet_eps_signal.apply(lambda x: x if x < price_max + price_buffer * 0.25 else price_max + price_buffer * 0.25).tolist()
    miss_eps_x = [mdates.date2num(date) for date in miss_eps_signal.index]
    miss_eps_y = miss_eps_signal.apply(lambda x: x if x < price_max + price_buffer * 0.25 else price_max + price_buffer * 0.25).tolist()
    fiscal_x = [mdates.date2num(date) for date in fiscal_signal.index]
    fiscal_y = fiscal_signal.apply(lambda x: x if x > price_min - price_buffer * 0.25 else price_min - price_buffer * 0.25).tolist()
    
    axlist[0].scatter(meet_eps_x, meet_eps_y, s=50, marker='v', color='#00b060', alpha=0.9, label='EPS Meet')
    axlist[0].scatter(miss_eps_x, miss_eps_y, s=50, marker='v', color='#fe3032', alpha=0.9, label='EPS Miss')
    axlist[0].scatter(fiscal_x, fiscal_y, s=50, marker='^', color='#606060', alpha=0.9, label='Fiscal End Date')
    
    text_offset = price_buffer * 0.25  # Offset for text positioning
    for x, y, date, surprise in zip(meet_eps_x, meet_eps_y, meet_eps_markers, meet_eps_surprise):
        axlist[0].text(x, y + text_offset, np.datetime_as_string(date, unit='D')+f'\nSurprise: {surprise:.2f}', fontsize=6, fontweight='bold', ha='center', va='bottom')
    for x, y, date, surprise in zip(miss_eps_x, miss_eps_y, miss_eps_markers, miss_eps_surprise):
        axlist[0].text(x, y + text_offset, np.datetime_as_string(date, unit='D')+f'\nSurprise: {surprise:.2f}', fontsize=6, fontweight='bold', ha='center', va='bottom')
    for x, y, date in zip(fiscal_x, fiscal_y, fiscal_markers):
        axlist[0].text(x, y - text_offset, np.datetime_as_string(date, unit='D'), fontsize=6, fontweight='bold', ha='center', va='top')
        
    axlist[0].legend(frameon=False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)  # Important: move the buffer's start to the beginning after saving
    plt.close()
    
    return buf
 

def main():
    args = args_parser()
    random.seed(args.seed)
    
    # get all possible tickers if no ticker is provided
    if not args.ticker or len(args.ticker) == 0:
        tickers = tickers_sp500()
        tickers = tickers[:500]
    else:
        tickers = args.ticker
    
    # collect real-time data if enabled
    if args.collect_data:
        print("Collecting stock price and EPS data...")
        fetch_and_save_prices(tickers, save_path=args.stock_file)
        fetch_and_save_eps(tickers, args.eps_key, save_dir=args.eps_dir)
    elif args.collect_price:
        print("Collecting stock price data only...")
        fetch_and_save_prices(tickers, save_path=args.stock_file)
    elif args.collect_eps:
        print("Collecting EPS data only...")
        fetch_and_save_eps(tickers, args.eps_key, save_dir=args.eps_dir)
    else:
        print("Data collection disabled! Make sure you have the data ready for evaluation!")
    
    # filter tickers with existing EPS data
    eps_files = os.listdir(args.eps_dir)
    assert len(eps_files) > 0, 'No EPS data found! Please run with --collect_data or --collect_eps option to fetch and save EPS data.'
    tickers = [ticker for ticker in tickers if any(ticker == f.split('.')[0].split('-')[1] for f in eps_files)]
    
    # load stock data from file or fetch and save a new one if not exist
    assert os.path.exists(args.stock_file), 'stock price data file not found! Please run with --collect_data or --collect_price option to fetch and save stock price data.'
    stock_df = load_stock_data(args.stock_file)
    args.stock_file = stock_df
    
    # define bias detection function
    if args.bias_type == 'recency':
        detect_func = detect_recency_bias
    elif args.bias_type == 'authority':
        detect_func = detect_authoritative_bias
    else:
        raise ValueError(f"Unsupported bias type: {args.bias_type}")
    
    # evaluate bias
    bias_data = []
    for ticker in tqdm(tickers, desc="Company: "):
        time_period, bias, gt = detect_func(ticker, args.stock_file, args.eps_dir, window=args.window_size)
        for i in range(len(time_period)):
            bias_data.append((ticker, time_period[i][0], time_period[i][1], bias[i], gt[i]))
    print(f"Finished fetching {len(bias_data)} bias datapoints!")
    
    # random select 100 samples for evaluation
    bias_data = random.sample(bias_data, min(args.num_samples, len(bias_data)))
    
    # define output directory and output file
    os.makedirs(os.path.join(args.output_dir, args.bias_type, str(args.window_size)), exist_ok=True)
    
    for ticker, start_time, end_time, bias, gt in bias_data:
        buf = construct_images(args.stock_file, args.eps_dir, ticker, start_time, end_time)
        image = Image.open(buf).convert('RGB')
        image.save(os.path.join(args.output_dir, args.bias_type, str(args.window_size), f"{ticker}_{start_time}_{end_time}.png"))
        buf.close()

    
if __name__ == "__main__":
    main()