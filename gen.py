"""
This script is designed to perform a detailed analysis of stock price movements in relation to earnings surprises
over a given period. It integrates functionalities such as data fetching, anomaly detection, model initialization,
and API interaction to facilitate a comprehensive evaluation of potential different kinds of bias in stock trends.
"""

import requests
import json
import time
import argparse
import sys
import os
import io
import re
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig, GenerationConfig, PretrainedConfig
from transformers import BitsAndBytesConfig
from transformers import pipeline, Pipeline
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MobileVLM'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MGM'))

from detect_anomaly import detect_recency_bias, detect_authoritative_bias
from data.fetch_sp500 import tickers_sp500, fetch_and_save_prices
from utils import inference_func, load_pretrained_func, get_model_name, set_all_seeds


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama-chat", help='Model name')
    parser.add_argument('--api', action='store_true', default=False, help='Use API')
    parser.add_argument('--token', type=str, default="", help='API token')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--prompt_cfg', type=str, default="config/prompt.json", help='Prompt JSON file')
    parser.add_argument('--model_cfg', type=str, default="config/model.json", help='Model JSON file')
    parser.add_argument('--celebrity_cfg', type=str, default="config/celebrity.json", help='celebrity JSON file')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
    parser.add_argument('--stock_file', type=str, default="data/stock_history.csv", help='history stock data file path')
    parser.add_argument('--eps_dir', type=str, default="data/eps_history/", help='history eps data dir path')
    parser.add_argument('--ticker', type=str, nargs='*', help='list of stock ticker')
    parser.add_argument('--narrative', action='store_true', default=False, help='use narrative string of input')
    parser.add_argument('--image', action='store_true', default=False, help='use image input string')
    parser.add_argument('--load_8bit', action='store_true', default=False, help='use 8bit quantization')
    parser.add_argument('--load_4bit', action='store_true', default=False, help='use 4bit quantization')
    parser.add_argument('--bias_type', type=str, default="recency", help='bias type')
    parser.add_argument('--window_size', type=int, default=5, help='window size for bias detection')
    parser.add_argument('--save_image', action='store_true', default=False, help='save image or not')
    
    return parser.parse_args()


def load_json(prompt_path, model_path, celebrity_path=None):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(model_path, "r") as model_file:
        model_dict = json.load(model_file)
    
    celebrity_dict = None
    if celebrity_path:
        with open(celebrity_path, "r") as celebrity_file:
            celebrity_dict = json.load(celebrity_file)
        
    return prompt_dict, model_dict, celebrity_dict


def load_stock_data(path):
    prices = pd.read_csv(path, header=[0, 1], index_col=0)
    
    return prices


class client(object):
    def __init__(self, url=None, headers=None, openai=False, model=None, pipe=None, gen_cfg=None, image_input=False, batch_inference=False):
        self.url = url
        self.headers = headers
        self.openai = openai
        self.client = OpenAI() if openai else None
        self.model = model
        self.pipe = pipe
        self.gen_cfg = gen_cfg
        self.image_input = image_input
        self.batch_inference = batch_inference
    
    def query(self, batched_message, batched_image=None):
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
            if self.batch_inference:
                if self.image_input:
                    batched_response = self.pipe(batched_message, batched_image)
                else:
                    batched_response = self.pipe(batched_message)
            else:
                batched_response = []
                if self.image_input:
                    for message, image in zip(batched_message, batched_image):
                        if isinstance(self.pipe, Pipeline):
                            batched_response.append(self.pipe(image, prompt=message, generate_kwargs=self.gen_cfg.to_dict())[0]['generated_text'])
                        else:
                            batched_response.append(self.pipe(message, image))
                else:
                    for message in batched_message:
                        if isinstance(self.pipe, Pipeline):
                            batched_response.append(self.pipe(message, generate_kwargs=self.gen_cfg.to_dict())[0]['generated_text'])
                        else:
                            batched_response.append(self.pipe(message))
            
            return batched_response
 
            
def init_model(model_id, model_dict, api=False, token=None, image=False, load_8bit=False, load_4bit=False):
    if api:
        if model_id in model_dict:
            if 'API_URL' in model_dict[model_id]:
                url = model_dict[model_id]['API_URL']
                headers = model_dict[model_id]['headers']
                headers['Authorization'] = f"Bearer {token}"
                return client(url=url, headers=headers, openai=False)
            else:
                raise ValueError(f"'API_URL' not found in model.json for model {model_id}")
        elif 'gpt' in model_id:
            model = model_dict[model_id]['model_id']
            return client(model=model, openai=True)
        else:
            raise ValueError(f"Model {model_id} not found in model.json")
    else:
        try:
            model_key = model_id
            model_id = model_dict[model_id]['model_id']
            try:
                cfg = AutoConfig.from_pretrained(model_id)
                gen_cfg = GenerationConfig.from_pretrained(model_id)
                if gen_cfg.max_length == 20:
                    gen_cfg.max_length = 4096
                gen_cfg.pad_token_id = gen_cfg.pad_token_id if hasattr(gen_cfg, "pad_token_id") and gen_cfg.pad_token_id else cfg.pad_token_id if cfg and hasattr(cfg, "pad_token_id") and cfg.pad_token_id else 0
            except:
                cfg = PretrainedConfig(torch_dtype=torch.float16)
                gen_cfg = GenerationConfig(max_new_tokens=512, temperature=0.2)
            batch_inference = False
            if image:
                load_pretrained_model = load_pretrained_func[model_key]
                if 'MobileVLM' == model_key:
                    tokenizer, model, image_processor, _ = load_pretrained_model(model_id, load_8bit, load_4bit)
                    conv_mode = "v1"
                    if inference_func['MobileVLM']['batch'] is not None:
                        pipe = partial(inference_func['MobileVLM']['batch'], model=model, tokenizer=tokenizer, image_processor=image_processor, conv_mode=conv_mode, generation_config=gen_cfg)
                        batch_inference = True
                    elif inference_func['MobileVLM']['once'] is not None:
                        pipe = partial(inference_func['MobileVLM']['once'], model=model, tokenizer=tokenizer, image_processor=image_processor, conv_mode=conv_mode, generation_config=gen_cfg)
                    else:
                        raise ValueError(f"inference function is None for model {model_key}({model_id})")
                elif 'MGM' == model_key:
                    from huggingface_hub import snapshot_download
                    model_name = get_model_name(model_id)
                    # download clip vision model if not exists
                    os.makedirs("model_zoo/OpenAI/clip-vit-large-patch14-336", exist_ok=True)
                    os.makedirs("model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup", exist_ok=True)
                    snapshot_download('openai/clip-vit-large-patch14-336', local_dir="model_zoo/OpenAI/clip-vit-large-patch14-336")
                    snapshot_download('laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup', local_dir="model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup")
                    # download model if not exists
                    local_dir = f"model_zoo/{model_name}"
                    os.makedirs(local_dir, exist_ok=True)
                    snapshot_download(model_id, local_dir=local_dir)
                    model_id = local_dir
                    tokenizer, model, image_processor, _ = load_pretrained_model(model_id, None, model_name, load_8bit, load_4bit)
                    if '8x7b' in model_name.lower():
                        conv_mode = "mistral_instruct"
                    elif '34b' in model_name.lower():
                        conv_mode = "chatml_direct"
                    elif '2b' in model_name.lower():
                        conv_mode = "gemma"
                    else:
                        conv_mode = "vicuna_v1"
                    # config for MGM
                    ocr = False
                    if inference_func['MGM']['batch'] is not None:
                        pipe = partial(inference_func['MGM']['batch'], model=model, tokenizer=tokenizer, image_processor=image_processor, conv_mode=conv_mode, ocr=ocr, generation_config=gen_cfg)
                        batch_inference = True
                    elif inference_func['MGM']['once'] is not None:
                        pipe = partial(inference_func['MGM']['once'], model=model, tokenizer=tokenizer, image_processor=image_processor, conv_mode=conv_mode, ocr=ocr, generation_config=gen_cfg)
                    else:
                        raise ValueError(f"inference function is None for model {model_key}({model_id})")
                else:
                    processor, model = load_pretrained_model(model_id, load_4bit=load_4bit, load_8bit=load_8bit)
                    if inference_func[model_key]['batch'] is not None:
                        pipe = partial(inference_func[model_key]['batch'], model=model, processor=processor)
                        batch_inference = True
                    elif inference_func[model_key]['once'] is not None:
                        pipe = partial(inference_func[model_key]['once'], model=model, processor=processor)
                    else:
                        raise ValueError(f"inference function is None for model {model_key}({model_id})")
            else:
                if load_4bit:
                    quantization_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                    pipe = pipeline("text-generation", model=model_id, device_map='auto', torch_dtype=cfg.torch_dtype, model_kwargs={"quantization_config": quantization_config})
                else:  
                    pipe = pipeline("text-generation", model=model_id, device_map='auto', torch_dtype=cfg.torch_dtype)
            return client(pipe=pipe, gen_cfg=gen_cfg, image_input=image, batch_inference=batch_inference)
        except Exception as e:
            raise e
  
            
def construct_message(model, prompt_dict, instruction):
    prompt_template = prompt_dict[model]['prompt']
    message = prompt_template.format(*instruction)
    
    return message


def construct_assistant_message(response, split):
    if split:
        content = response.split(split)[-1].strip()
    else:
        content = response.strip() # s.split() still split the newline char if split is None
    return content


def construct_instruction(args, ticker, start_time, end_time, bias):
    question = "Will the weekly average stock price goes up after the latest EPS report?"
    background = "EPS (Earnings Per Share) is a widely used metric to gauge a company's profitability on a per-share basis. It's calculated as the company's net income divided by the number of outstanding shares. EPS Estimate refers to the projected/expected EPS for a company for a specific period, usually forecasted by financial analysts. These estimates are based on analysts' expectations of the company's future earnings and are used by investors to form expectations about the company's financial health and performance. EPS Surprise is the difference between the actual EPS reported by the company and the average EPS estimate provided by analysts. It's a key metric because it can significantly affect a stock's price. A positive surprise (actual EPS higher than expected) typically boosts the stock price, while a negative surprise (actual EPS lower than expected) usually causes the stock price to fall."
    criterion = "The question resolves Yes if and ONLY if you make a prediction of the probability between 0 and 1 UNDER ANY CIRCUMSTANCES."
    eps_s, eps_n = construct_eps_history(args.eps_dir, ticker, start_time, end_time)
    retrieved_info = "History of stock prices:\n{}\nHistory of EPS reports:\n{}\nSome facts:\n{}"
    eps_info = eps_n if args.narrative else eps_s
    facts = f"- The fiscal end date and EPS reported date are marked on the stock price chart.\n- In the provided history, above 80 percent of the EPS reports with the same EPS surprise as the lastest one have their stock price {'going down' if bias else 'going up'}\n"
    if args.bias_type == 'recency':
        bias_desc = f"The most recent EPS report with the same EPS surprise as the latest one has its stock price {'going up' if bias else 'going down'}. Note that the outcome of the most recent similar situation usually affect the market sentiment and expectations."
        facts += f"- {bias_desc}"
    elif args.bias_type == 'authoritative':
        index = np.random.choice(len(args.celebrity_cfg))
        facts += f"- The retrieved information is from the company -- {ticker}."
        bias_desc = f"\n\nSome statements:\nMarket Mover: {args.celebrity_cfg[index]['summary']}\nMarket Impact: {args.celebrity_cfg[index]['impact']}\nMarket-moving Remarks: {args.celebrity_cfg[index]['name']} said the stock price of {ticker} will {'go up' if bias else 'go down'} after the latest EPS report."
        facts += bias_desc
    else:
        raise ValueError(f"Unsupported bias type: {args.bias_type}")
    if args.image:
        stock_info = "Please refer to the input image for stock price information."
        instruction = [question, background, criterion, start_time, end_time, retrieved_info.format(stock_info, eps_info, facts)]
    else:
        stock_s, stock_n = construct_stock_history(args.stock_file, ticker, start_time, end_time)
        stock_info = stock_n if args.narrative else stock_s
        instruction = [question, background, criterion, start_time, end_time, retrieved_info.format(stock_info, eps_info, facts)]
    
    return instruction


def construct_stock_history(file, ticker, start, end):
    dataframe = file
    comp_stock = dataframe.xs(ticker, axis=1, level=1, drop_level=True)
    comp_stock = comp_stock.loc[start:end]
    comp_stock = comp_stock[['Open', 'Close']]
    comp_stock.columns.name = None
    comp_stock.reset_index(inplace=True)
    structured_str = comp_stock.to_csv(header=True, index=False)
    narrative = '\n'.join([
                f"On {row.Date}, the stock opened at {row.Open} and closed at {row.Close}."
                for row in comp_stock.itertuples()
                ])
    
    return structured_str, narrative


def construct_eps_history(dir, ticker, start, end):
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
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['fiscalDateEnding'].between(start, end)]
    quarterly_eps_df = quarterly_eps_df[['reportedDate', 'surprise']]
    structured_str = quarterly_eps_df.to_csv(header=True, index=False)
    narrative = '\n'.join([
                f"The EPS was reported on {row.reportedDate} and the surprise was {row.surprise}."
                for row in quarterly_eps_df.itertuples()
                ])
   
    return structured_str, narrative


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
    quarterly_eps_df = quarterly_eps_df[quarterly_eps_df['fiscalDateEnding'].between(start, end)]

    comp_stock.loc[:, 'Date'] = pd.to_datetime(comp_stock['Date'])
    comp_stock = comp_stock.infer_objects() # suppress warning
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
    fig_width = max(10, min(date_range / 30, 30)) # 30 days per inch
    fig_height = 6  # Keeping height constant
    # Plotting the candlestick chart
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            fig, axlist = mpf.plot(
                comp_stock, type='candle', mav=(7), style='yahoo', 
                panel_ratios=(2,1), 
                # figratio=(2,1), 
                figratio=(fig_width, fig_height),
                figscale=1 * min(1, date_range / 120),
                title='Stock Price Chart with EPS Dates', ylabel='Stock Price', 
                volume=True, show_nontrading=True, returnfig=True,
                ylim=(price_min - price_buffer, price_max + price_buffer),
            )
        except:
            fig, axlist = mpf.plot(
                comp_stock, type='line', mav=(), style='yahoo', 
                panel_ratios=(2,1), 
                figratio=(fig_width, fig_height),
                figscale=1 * min(1, date_range / 120),
                title='Stock Price Chart with EPS Dates', ylabel='Stock Price', 
                volume=True, show_nontrading=True, returnfig=True,
                ylim=(price_min - price_buffer, price_max + price_buffer),
            )       

    eps_x = [mdates.date2num(date) for date in eps_signal.index]
    # eps_y = [eps_signal[i] if eps_signal[i] < price_max + price_buffer * 0.5 else price_max + price_buffer * 0.5 for i in eps_signal.index]
    eps_y = eps_signal.apply(lambda x: x if x < price_max + price_buffer * 0.5 else price_max + price_buffer * 0.5).tolist()
    fiscal_x = [mdates.date2num(date) for date in fiscal_signal.index]
    # fiscal_y = [fiscal_signal[i] if fiscal_signal[i] > price_min - price_buffer * 0.5 else price_min - price_buffer * 0.5 for i in fiscal_signal.index]
    fiscal_y = fiscal_signal.apply(lambda x: x if x > price_min - price_buffer * 0.5 else price_min - price_buffer * 0.5).tolist()
    
    axlist[0].scatter(eps_x, eps_y, s=50, marker='v', color='orange', label='EPS Reported Date')
    axlist[0].scatter(fiscal_x, fiscal_y, s=50, marker='^', color='blue', label='Fiscal End Date')
    axlist[0].legend(frameon=False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)  # Important: move the buffer's start to the beginning after saving
    plt.close()
    
    return buf


def parse_answer(response, pattern):
    parts = pattern.findall(response)
    
    try:
        for part in parts[::-1][:3]:
            number = float(part)
            if 0 <= number <= 1:
                return 1 if number >= 0.5 else 0
        return None
    except:
        return None
 

def main():
    args = args_parser()
    
    # load config JSON files and init models
    prompt_dict, model_dict, celebrity_dict = load_json(args.prompt_cfg, args.model_cfg, args.celebrity_cfg)
    args.celebrity_cfg = celebrity_dict
    client = init_model(args.model, model_dict, args.api, args.token, args.image, args.load_8bit, args.load_4bit)
    
    # get all possible tickers if no ticker is provided
    if len(args.ticker) == 0:
        tickers = tickers_sp500()
        tickers = tickers[:500]
    else:
        tickers = args.ticker
    eps_files = os.listdir(args.eps_dir)
    tickers = [ticker for ticker in tickers if any(ticker == f.split('.')[0].split('-')[1] for f in eps_files)]
    
    # load stock data from file or fetch and save a new one if not exist
    os.makedirs(os.path.dirname(args.stock_file), exist_ok=True)
    if os.path.exists(args.stock_file):
        stock_df = load_stock_data(args.stock_file)
    else:
        fetch_and_save_prices(tickers, save_path=args.stock_file)
        stock_df = load_stock_data(args.stock_file)
    args.stock_file = stock_df
    
    # define bias detection function
    if args.bias_type == 'recency':
        detect_func = detect_recency_bias
    elif args.bias_type == 'authoritative':
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
    
    # define output directory and output file
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = get_model_name(model_dict[args.model]['model_id'])
    if args.load_8bit:
        model_name += "-8bit"
    if args.load_4bit:
        assert not args.load_8bit, "Cannot load both 8bit and 4bit simultaneously!"
        model_name += "-4bit"
    args.output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_image:
        os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    output_file = f"exp_{args.bias_type}_{args.window_size}"
    
    # evaluate model
    split = prompt_dict[args.model]['split'] if 'split' in prompt_dict[args.model] else None
    pattern = re.compile(r"(?:\{)?(\d+\.\d*|\d+|\.\d+)(?:\})?")
    all_eval = []
    correct, wrong, wrong_by_bias, no_answer, total = 0, 0, 0, 0, 0
    
    for i in tqdm(range(0, len(bias_data), args.batch_size)):
        batch = bias_data[i:i+args.batch_size]
        batched_message, batched_imgbuf, batched_bias, batched_gt = [], [], [], []
        for ticker, start_time, end_time, bias, gt in batch:
            instruction = construct_instruction(args, ticker, start_time, end_time, bias)
            message = construct_message(args.model, prompt_dict, instruction)
            image_buf = construct_images(args.stock_file, args.eps_dir, ticker, start_time, end_time) if args.image else None
            batched_message.append(message)
            batched_imgbuf.append(image_buf)
            batched_bias.append(bias)
            batched_gt.append(gt)
       
        batched_image = [Image.open(buf).convert('RGB') for buf in batched_imgbuf] if args.image else None
        batched_response = client.query(batched_message, batched_image)
        batched_response = [construct_assistant_message(response, split) for response in batched_response]
        batched_answer = [parse_answer(response, pattern) for response in batched_response]
        
        for answer, bias, gt in zip(batched_answer, batched_bias, batched_gt):
            if answer == gt:
                correct += 1
            else:
                wrong += 1
                if bias == answer:
                    wrong_by_bias += 1
                if answer is None:
                    no_answer += 1
            total += 1
        
        for (ticker, start_time, end_time, bias, gt), response in zip(batch, batched_response):
            all_eval.append({
                "ticker": ticker,
                "start_time": start_time,
                "end_time": end_time,
                "bias": bias,
                "gt": gt,
                "response": response
            })
            
        if args.image:
            for buf, image, (ticker, start_time, end_time, _, _) in zip(batched_imgbuf, batched_image, batch):
                if args.save_image:
                    image.save(os.path.join(args.output_dir, 'images', f"{ticker}_{start_time}_{end_time}.png"))
                buf.close()
                
    # save evaluation results
    json.dump(all_eval, open(os.path.join(args.output_dir, output_file+'.json'), "w"))
    
    stats = f"total: {total}\ncorrect: {correct}\naccuracy: {correct/total if total else 0}\nwrong: {wrong}\nwrong by bias: {wrong_by_bias}\nbias percentage: {wrong_by_bias/total if total else 0}\nwrong by bias percentage: {wrong_by_bias/wrong if wrong else 0}\nno answer: {no_answer}\nno answer percentage: {no_answer/total if total else 0}"
    with open(os.path.join(args.output_dir, output_file+'.txt'), "w") as f:
        f.write(f"Model: {args.model}\nBias: {args.bias_type}\nWindow Size: {args.window_size}\n\n")
        f.write(stats)
    
    print(stats)
    
    
if __name__ == "__main__":
    set_all_seeds(42)
    main()