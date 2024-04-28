"""
This script is designed to perform a detailed analysis of stock price movements in relation to earnings surprises
over a given period. It integrates functionalities such as data fetching, anomaly detection, model initialization,
and API interaction to facilitate a comprehensive evaluation of potential different kinds of bias in stock trends.
"""

import requests
import json
import time
import argparse
import os
import io
import re
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, GemmaTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForPreTraining, AutoModel
from transformers import AutoConfig, GenerationConfig
from transformers import BitsAndBytesConfig
from transformers import pipeline
import accelerate

import torch

from detect_anomaly import detect_recency_bias
from data.fetch_sp500 import tickers_sp500, fetch_and_save_prices


def args_parser():
    """
    Parses command-line arguments to configure the script's execution parameters. This function defines several
    options to customize the behavior of the script, such as choosing the model, whether to use an API, batch sizes,
    and various file paths for input and output.

    Arguments:
        --model (str): Specifies the model name (defined in prompt config .json file) to be used, defaulting to 'llama-chat'.
        --api (bool): If set, indicates that the script should use an API for inference, default is False.
        --token (str): Provides the API token if API usage is enabled, default is an empty string.
        --batch_size (int): Sets the batch size for processing, default is 4.
        --prompt_cfg (str): Path to the JSON configuration file for prompts, default is 'config/prompt.json'.
        --model_cfg (str): Path to the JSON configuration file for model settings, default is 'config/model.json'.
        --output_dir (str): Directory where output files will be stored, default is 'output'.
        --stock_file (str): Path to the historical stock data file, default is 'data/stock_history.csv'.
        --eps_dir (str): Directory path for historical earnings per share data, default is 'data/eps_history/'.
        --ticker (list of str): Allows multiple stock tickers to be specified, captures all arguments that appear.
        --narrative (bool): If set, uses a narrative form of historical data, default is False (structured form of historical data).
        --image (bool): If set, uses image inputs along with text, default is False.
        --quant (bool): If set, enables quantization for model processing, default is False.

    Returns:
        argparse.Namespace: Returns an object containing the parsed command-line arguments. Each argument is accessible
        as an attribute of this object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama-chat", help='Model name')
    parser.add_argument('--api', action='store_true', default=False, help='Use API')
    parser.add_argument('--token', type=str, default="", help='API token')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--prompt_cfg', type=str, default="config/prompt.json", help='Prompt JSON file')
    parser.add_argument('--model_cfg', type=str, default="config/model.json", help='Model JSON file')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
    parser.add_argument('--stock_file', type=str, default="data/stock_history.csv", help='history stock data file path')
    parser.add_argument('--eps_dir', type=str, default="data/eps_history/", help='history eps data dir path')
    parser.add_argument('--ticker', type=str, nargs='*', help='list of stock ticker')
    parser.add_argument('--narrative', action='store_true', default=False, help='use narrative string of input')
    parser.add_argument('--image', action='store_true', default=False, help='use image input string')
    parser.add_argument('--quant', action='store_true', default=False, help='use quantization')
    
    return parser.parse_args()


def load_json(prompt_path, model_path):
    """
    Loads JSON configuration files from specified paths. This function is designed to read two separate JSON files:
    one containing prompt configurations and another containing model configurations. Both files are expected to be
    in standard JSON format.

    Parameters:
    - prompt_path (str): The file path to the JSON file containing prompt configurations. This file should define
      various prompts that might be used throughout the script.
    - model_path (str): The file path to the JSON file containing model configurations. This file should define
      settings and parameters related to the models used in the script.

    Returns:
    - tuple: Returns a tuple containing two dictionaries:
        - prompt_dict (dict): A dictionary containing the configurations loaded from the prompt JSON file.
        - model_dict (dict): A dictionary containing the configurations loaded from the model JSON file.
    """
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(model_path, "r") as model_path:
        model_dict = json.load(model_path)

    return prompt_dict, model_dict


def load_stock_data(path):
    """
    Loads stock price data from yfinance stock history CSV file specified by the path. The function expects the CSV file to have a
    multi-level header in the first two rows and the first column used as the index, typically representing dates.

    Parameters:
    - path (str): The file path to the CSV containing the stock price data. This path should point to a properly
      formatted CSV file with multi-level headers, which often include stock identifiers and price attributes like
      'Open', 'Close', 'High', 'Low', etc.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the loaded stock price data. The DataFrame will have a
      multi-index header and dates as the index, facilitating easy manipulation and analysis of stock prices over time.
    """
    prices = pd.read_csv(path, header=[0, 1], index_col=0)
    
    return prices


class client(object):
    """
    A client class to handle requests to APIs or local model pipelines for text-only or multimodal predictions. 
    It abstracts the complexities of different backends such as OpenAI's GPT models, Hugging Face models, or other custom API endpoints.

    Attributes:
    - url (str): The URL endpoint for custom APIs. Used if not interacting with OpenAI's API.
    - headers (dict): Headers to be sent with requests, typically including authorization tokens.
    - openai (bool): Flag to determine if this client is using OpenAI's API.
    - model (str): The model identifier specified in model config file, e.g. 'config/model.json'.
    - pipe (any): A Hugging Face pipeline for local model.
    - gen_cfg (any): Configuration for generation tasks, used with local pipelines.
    - image_input (bool): Flag to indicate if the client supports image input data for queries.

    Methods:
    - query(batched_message, batched_image=None): Sends a batch of messages, and optionally images, to the configured
      backend (API or local model) and returns the responses.
    """
    def __init__(self, url=None, headers=None, openai=False, model=None, pipe=None, gen_cfg=None, image_input=False):
        self.url = url
        self.headers = headers
        self.openai = openai
        self.client = OpenAI() if openai else None
        self.model = model
        self.pipe = pipe
        self.gen_cfg = gen_cfg
        self.image_input = image_input
    
    def query(self, batched_message, batched_image=None):
        """
        Processes a batch of messages and optionally images through the configured backend (API or local model).

        Parameters:
        - batched_message (list of str): List of text messages to be processed.
        - batched_image (list of PIL.Image or similar, optional): List of image data, used if `image_input` is True.

        Returns:
        - list of str: Responses from the backend for each input message or image.
        """
        if self.openai:
            if batched_image is not None:
                raise NotImplementedError("Image input is not implemented for OpenAI API")
            
            batched_response = []
            for message in batched_message:
                retry = 3
                while retry > 0:
                    try:
                        completion = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": message}],
                        )
                    except:
                        retry -= 1
                        completion = None
                        time.sleep(10)
                
                batched_response.append(completion.choices[0].message.content) if completion else batched_response.append("Missing")
            
            return batched_response
        
        elif not self.pipe:
            if batched_image is not None:
                raise NotImplementedError("Vision2Seq task is not implemented for Hugging Face Inference API")
            
            batched_response = []
            for message in batched_message:
                retry = 3
                while retry > 0:
                    try:
                        payload = {"inputs": message}
                        response = requests.post(self.url, headers=self.headers, json=payload)
                    except:
                        retry -= 1
                        response = None
                        time.sleep(10)
                    
                batched_response.append(response.json()[0]['generated_text']) if response else batched_response.append("Missing")
            
            return batched_response
        
        else: 
            if self.image_input and batched_image is not None:
                batched_response = []
                for message, image in zip(batched_message, batched_image):
                    batched_response.append(self.pipe(image, prompt=message, generate_kwargs=self.gen_cfg.to_dict())[0]['generated_text'])
            else:
                batched_response = [response[0]['generated_text'] for response in self.pipe(batched_message, generation_config=self.gen_cfg, clean_up_tokenization_spaces=True)]
            
            return batched_response
 
            
def init_model(model_id, model_dict, api=False, token=None, image=False, quant=False):
    """
    Initializes and configures a model client for API or local model inference based on specified parameters.
    This function can set up clients for both external APIs and local inference pipelines, supporting
    optional features such as image input and quantization.

    Parameters:
    - model_id (str): The identifier for the model to be initialized.
    - model_dict (dict): A dictionary loaded from model config JSON file.
    - api (bool, optional): Specifies whether to use a model API. Defaults to False.
    - token (str, optional): The API token for authentication if using an API. Defaults to None.
    - image (bool, optional): Indicates whether the model should support image inputs. Defaults to False.
    - quant (bool, optional): Indicates whether to enable quantization for the model. Defaults to False.

    Returns:
    - client: An instance of the `client` class configured to interact with the specified model, either through
      an API or a local inference pipeline.
    """
    if api:
        if model_id in model_dict:
            if 'API_URL' in model_dict[model_id]:
                url = model_dict[model_id]['API_URL']
                headers = model_dict[model_id]['HEADERS']
                headers['Authorization'] = f"Bearer {token}"
                return client(url=url, headers=headers, openai=False)
            else:
                model = model_dict[model_id]['model_id']
                return client(model=model, openai=True)
        else:
            raise ValueError(f"Model {model_id} not found in model.json")
    else:
        try:
            model_id = model_dict[model_id]['model_id']
            cfg = AutoConfig.from_pretrained(model_id)
            gen_cfg = GenerationConfig.from_pretrained(model_id)
            if gen_cfg.max_length == 20:
                gen_cfg.max_length = 4096*2
            gen_cfg.pad_token_id = gen_cfg.pad_token_id if hasattr(gen_cfg, "pad_token_id") and gen_cfg.pad_token_id else \
            cfg.pad_token_id if hasattr(cfg, "pad_token_id") and cfg.pad_token_id else 0
            if quant:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            if image:
                if quant:
                    pipe = pipeline("image-to-text", model=model_id, device_map='auto', torch_dtype=cfg.torch_dtype, model_kwargs={"quantization_config": quantization_config})
                else:  
                    pipe = pipeline("image-to-text", model=model_id, device_map='auto', torch_dtype=cfg.torch_dtype)        
            else:
                if quant:
                    pipe = pipeline("text-generation", model=model_id, device_map='auto', torch_dtype=cfg.torch_dtype, model_kwargs={"quantization_config": quantization_config})
                else:  
                    pipe = pipeline("text-generation", model=model_id, device_map='auto', torch_dtype=cfg.torch_dtype)
            return client(pipe=pipe, gen_cfg=gen_cfg, image_input=image)
        except Exception as e:
            raise e
  
            
def construct_message(model, prompt_dict, instruction):
    """
    Constructs a message based on a predefined prompt template and specific instructions provided. This function
    formats a template string by inserting the provided instruction elements into placeholders in the template.

    Parameters:
    - model (str): The identifier for the model, which corresponds to a specific key in the `prompt_dict` dictionary.
    - prompt_dict (dict): A dictionary loaded from prompt config JSON file.
    - instruction (list of str): A list of string elements that are used to fill in the placeholders in the prompt template.

    Returns:
    - str: The fully constructed message that has been formatted with the specific instructions.
    """
    prompt_template = prompt_dict[model]['prompt']
    message = prompt_template.format(*instruction)
    
    return message


def construct_assistant_message(response, split):
    """
    Constructs an assistant message for multi-turn char models.
    """
    pass


def construct_instruction(args, ticker, start_time, end_time):
    """
    Constructs a structured instruction set for querying or processing based on historical stock price and EPS data.
    The function combines provided arguments with internally generated text components to formulate a complete
    instruction that describes a task related to stock trend prediction based on EPS surprises.

    Parameters:
    - args (Namespace): Command-line arguments.
    - ticker (str): The stock ticker symbol for which the instruction is being constructed.
    - start_time (str): The starting date/time for the data period of interest.
    - end_time (str): The ending date/time for the data period of interest.

    Returns:
    - list of str: A list of string elements that together form a comprehensive instruction set. This list includes
      a question component, background information, criteria for analysis, and formatted stock and EPS data descriptions.
    """
    question = "Stock trending prediction based off historical stocks' price and EPS data."
    background = "EPS (Earnings Per Share) is a widely used metric to gauge a company's profitability on a per-share basis. It's calculated as the company's net income divided by the number of outstanding shares. EPS Estimate refers to the projected (or expected) EPS for a company for a specific period, usually forecasted by financial analysts. These estimates are based on analysts' expectations of the company's future earnings and are used by investors to form expectations about the company's financial health and performance. EPS Surprise is the difference between the actual EPS reported by the company and the average EPS estimate provided by analysts. It's a key metric because it can significantly affect a stock's price. A positive surprise (actual EPS higher than expected) typically boosts the stock price, while a negative surprise (actual EPS lower than expected) usually causes the stock price to fall."
    criterion = "According to the historical stock price and EPS data, predict the stock trending after the lastest EPS report date with the EPS surprise reported."
    stock_s, stock_n = construct_stock_history(args.stock_file, ticker, start_time, end_time)
    eps_s, eps_n = construct_eps_history(args.eps_dir, ticker, start_time, end_time)
    if args.narrative:
        instruction = [question, background, criterion, stock_n, eps_n]
    else:
        instruction = [question, background, criterion, stock_s, eps_s]
    
    return instruction


def construct_stock_history(file, ticker, start, end):
    """
    Retrieves and formats stock price history for a given ticker over a specified time range. This function
    extracts the stock's opening and closing prices from a DataFrame and provides both a structured string
    representation and a narrative description of the price history.

    Parameters:
    - file (pd.DataFrame): A DataFrame containing stock price data with a multi-level column header, where
      the second level of the header includes the ticker symbols.
    - ticker (str): The stock ticker symbol for which to retrieve and format history data.
    - start (str): The start date for the period over which to retrieve stock data, formatted as 'YYYY-MM-DD'.
    - end (str): The end date for the period, formatted similarly to the start date.

    Returns:
    - tuple (str, str): A tuple containing two strings:
        1. structured_str: A structured string representation of the stock price data, formatted as a table.
        2. narrative: A narrative description of the stock's daily opening and closing prices over the specified period.
    """
    dataframe = file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock = comp_stock.loc[start:end]
    comp_stock = comp_stock[['Open', 'Close']]
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    structured_str = comp_stock.to_string(header=True, index=False)
    narrative = '\n'.join([
                f"On {row.Date}, the stock opened at {row.Open} and closed at {row.Close}."
                for row in comp_stock.itertuples()
                ])
    
    return structured_str, narrative


def construct_eps_history(dir, ticker, start, end):
    """
    Retrieves and formats the earnings per share (EPS) history for a given ticker within a specified date range.
    This function searches for an EPS data file in the specified directory, loads the data, and then extracts
    relevant EPS records between the start and end dates. It returns both a structured string format and a
    narrative description of the EPS data.

    Parameters:
    - dir (str): The directory path where EPS data JSON files are stored.
    - ticker (str): The stock ticker symbol for which to retrieve EPS data.
    - start (str): The start date for the period over which to retrieve EPS data, formatted as 'YYYY-MM-DD'.
    - end (str): The end date for the period, formatted similarly to the start date.

    Returns:
    - tuple (str, str): A tuple containing two strings:
        1. structured_str: A structured string representation of the EPS data, formatted as a table.
        2. narrative: A narrative description of each EPS entry.
    """
    files = os.listdir(dir)
    file = None
    for f in files:
        if ticker in f:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['fiscalDateEnding'].between(start, end)]
    structured_str = quarterly_eps_df.to_string(header=True, index=False)
    narrative = '\n'.join([
                f"For the quarter ending {row.fiscalDateEnding}, the EPS was {row.reportedEPS} reported on {row.reportedDate} and the estimated EPS was {row.estimatedEPS}. The surprise was {row.surprise} with a percentage of {row.surprisePercentage}."
                for row in quarterly_eps_df.itertuples()
                ])
   
    return structured_str, narrative


def construct_images(file, dir, ticker, start, end):
    """
    Generates a visual representation (candlestick chart) of stock price movements alongside important EPS (Earnings
    Per Share) report dates within a specified period. This function integrates stock data and EPS data to highlight
    significant financial reporting dates directly on the chart.

    Parameters:
    - file (pd.DataFrame): A DataFrame containing historical stock data with a multi-level header.
    - dir (str): The directory path where EPS data files are stored.
    - ticker (str): The stock ticker symbol for which the image is being constructed.
    - start (str): The start date for the period over which to generate the image, formatted as 'YYYY-MM-DD'.
    - end (str): The end date for the period, formatted similarly to the start date.

    Returns:
    - io.BytesIO: A buffer containing the generated image data, which can be saved to a file or displayed directly.
    """
    dataframe = file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock = comp_stock.loc[start:end]
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    files = os.listdir(dir)
    file = None
    for f in files:
        if ticker in f:
            file = f
            break
    if not file:
        raise FileNotFoundError(f"No EPS data found for {ticker}")
    with open(os.path.join(dir,file), "r") as f:
        eps_dict = json.load(f)
    f.close()
    quarterly_eps = eps_dict['quarterlyEarnings']
    quarterly_eps_df = pd.DataFrame(quarterly_eps)
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['fiscalDateEnding'].between(start, end)]

    comp_stock.loc[:, 'Date'] = pd.to_datetime(comp_stock['Date'])
    comp_stock.set_index('Date', inplace=True)
    # Ensure quarterly_eps_df['reportedDate'] and quarterly_eps_df['fiscalDateEnding'] are in datetime format
    quarterly_eps_df['reportedDate'] = pd.to_datetime(quarterly_eps_df['reportedDate'])
    quarterly_eps_df['fiscalDateEnding'] = pd.to_datetime(quarterly_eps_df['fiscalDateEnding'])
    # Creating markers for EPS report date and fiscal end date
    eps_markers = quarterly_eps_df.loc[quarterly_eps_df['reportedDate'].between(start, end), 'reportedDate'].values
    fiscal_markers = quarterly_eps_df.loc[quarterly_eps_df['fiscalDateEnding'].between(start, end), 'fiscalDateEnding'].values
    # Calculate minimum and maximum prices for y-axis scaling
    price_min = comp_stock[['Low']].min().min()  # min of 'Low' prices
    price_max = comp_stock[['High']].max().max()  # max of 'High' prices
    price_range = price_max - price_min
    price_buffer = price_range * 0.1  # 10% price buffer on each side
    # Calculate adaptive offsets for markers
    offset = price_range * 0.25  # 25% of the price range
    # Creating EPS signals
    all_dates = pd.date_range(start, end, freq='D')
    low_prices = comp_stock['Low'].reindex(all_dates).interpolate(method='linear')
    high_prices = comp_stock['High'].reindex(all_dates).interpolate(method='linear')
    eps_signal = high_prices.loc[eps_markers] + offset
    fiscal_signal = low_prices.loc[fiscal_markers] - offset
    # Setting figure size dynamically based on the date range
    date_range = (dt.datetime.strptime(end, '%Y-%m-%d') - dt.datetime.strptime(start, '%Y-%m-%d')).days
    fig_width = max(10, date_range / 30) # 30 days per inch
    fig_height = 6  # Keeping height constant
    # Plotting the candlestick chart
    fig, axlist = mpf.plot(
        comp_stock, type='candle', mav=7, style='yahoo', 
        panel_ratios=(2,1), 
        # figratio=(2,1), 
        figratio=(fig_width, fig_height),
        figscale=1, 
        title='Stock Price Chart with EPS Dates', ylabel='Stock Price', 
        volume=True, show_nontrading=True, returnfig=True,
        ylim=(price_min - price_buffer, price_max + price_buffer),
    )

    eps_x = [mdates.date2num(date) for date in eps_signal.index]
    eps_y = [eps_signal[i] if eps_signal[i] < price_max + price_buffer * 0.5 else price_max + price_buffer * 0.5 for i in eps_signal.index]
    fiscal_x = [mdates.date2num(date) for date in fiscal_signal.index]
    fiscal_y = [fiscal_signal[i] if fiscal_signal[i] > price_min - price_buffer * 0.5 else price_min - price_buffer * 0.5 for i in fiscal_signal.index]

    axlist[0].scatter(eps_x, eps_y, s=50, marker='v', color='orange', label='EPS Reported Date')
    axlist[0].scatter(fiscal_x, fiscal_y, s=50, marker='^', color='blue', label='Fiscal End Date')
    axlist[0].legend(frameon=False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)  # Important: move the buffer's start to the beginning after saving
    
    return buf


def parse_answer(response, split, pattern):
    """
    Parses a textual response to extract and evaluate numerical information based on a given pattern and split criteria.
    This function is designed to split the response at a specified delimiter, apply a regular expression pattern to find
    numeric values, and then determine a binary outcome based on the last numeric value extracted.

    Parameters:
    - response (str): The textual response from which to extract the number.
    - split (str): The delimiter used to split the response string. The function will use the last segment after this split.
    - pattern (re.Pattern): A compiled regular expression pattern used to find all numeric strings in the selected segment.

    Returns:
    - int: Returns 1 if the last numeric value found is greater than or equal to 0.5, otherwise returns 0.
    - None: Returns None if no numeric value is found or if the extraction and conversion to float fail.
    """
    answer = response.split(split)[-1]
    parts = pattern.findall(answer)
    
    try:
        number = float(parts[-1])
        return 1 if number >= 0.5 else 0
    except:
        return None
 

def main():
    args = args_parser()
    
    # load config JSON files and init models
    prompt_dict, model_dict = load_json(args.prompt_cfg, args.model_cfg)
    client = init_model(args.model, model_dict, args.api, args.token, args.image, args.quant)
    
    # get all possible tickers if no ticker is provided
    if len(args.ticker) == 0:
        tickers = tickers_sp500()
        tickers = tickers[:500]
    else:
        tickers = args.ticker
    eps_files = os.listdir(args.eps_dir)
    tickers = [ticker for ticker in tickers if any(ticker in f for f in eps_files)]
    
    # load stock data from file or fetch and save a new one if not exist
    os.makedirs(os.path.dirname(args.stock_file), exist_ok=True)
    if os.path.exists(args.stock_file):
        stock_df = load_stock_data(args.stock_file)
    else:
        fetch_and_save_prices(tickers, save_path=args.stock_file)
        stock_df = load_stock_data(args.stock_file)
    args.stock_file = stock_df
    
    # evaluate recency bias
    recency_bias_data = []
    for ticker in tickers:
        time_period, last, gt = detect_recency_bias(ticker, args.stock_file, args.eps_dir, window=5)
        for i in range(len(time_period)):
            recency_bias_data.append((ticker, time_period[i][0], time_period[i][1], last[i], gt[i]))

    # define output directory and output file
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    output_file = f"exp_{len(os.listdir(args.output_dir))}"
    # evaluate model
    split = prompt_dict[args.model]['split'] if 'split' in prompt_dict[args.model] else None
    pattern = re.compile(r"(?:\{)?(\d+\.\d*|\d+|\.\d+)(?:\})?")
    all_eval = []
    correct, wrong, bias, total = 0, 0, 0, 0 
    
    for i in range(0, len(recency_bias_data), args.batch_size):
        batch = recency_bias_data[i:i+args.batch_size]
        batched_message, batched_imgbuf, batched_last, batched_gt = [], [], [], []
        for ticker, start_time, end_time, last, gt in batch:
            instruction = construct_instruction(args, ticker, start_time, end_time)
            instruction[-2] = 'refer to input image' if args.image else None
            message = construct_message(args.model, prompt_dict, instruction)
            image_buf = construct_images(args.stock_file, args.eps_dir, ticker, start_time, end_time) if args.image else None
            batched_message.append(message)
            batched_imgbuf.append(image_buf)
            batched_last.append(last)
            batched_gt.append(gt)
       
        batched_image = [Image.open(buf) for buf in batched_imgbuf] if args.image else None
        batched_response = client.query(batched_message, batched_image)
        batched_answer = [parse_answer(response, split, pattern) for response in batched_response]
        
        for answer, last, gt in zip(batched_answer, batched_last, batched_gt):
            if answer == gt:
                correct += 1
            else:
                wrong += 1
                if last == answer:
                    bias += 1
            total += 1
        
        for (ticker, start_time, end_time, last, gt), response in zip(batch, batched_response):
            all_eval.append({
                "ticker": ticker,
                "start_time": start_time,
                "end_time": end_time,
                "last": last,
                "gt": gt,
                "response": response
            })
            
        if args.image:
            for buf, image, (ticker, start_time, end_time, _, _) in zip(batched_imgbuf, batched_image, batch):
                image.save(os.path.join(args.output_dir, 'images', f"{ticker}_{start_time}_{end_time}.png"))
                buf.close()
                
    # save evaluation results
    json.dump(all_eval, open(os.path.join(args.output_dir, output_file+'.json'), "w"))
    
    stats = f"accuracy: {correct/total if total else 0}\nbias: {bias}\nbias percentage: {bias/total if total else 0}\nwrong by bias: {bias/wrong if wrong else 0}"
    with open(os.path.join(args.output_dir, output_file+'.txt'), "w") as f:
        f.write(stats)
    
    print(stats)
    
    
if __name__ == "__main__":
    main()