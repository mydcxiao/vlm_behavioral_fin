import requests
import json
import time
import argparse
import os
import io
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


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama-chat", help='Model name')
    parser.add_argument('--api', action='store_true', default=False, help='Use API')
    parser.add_argument('--token', type=str, default="", help='API token')
    parser.add_argument('--prompt_cfg', type=str, default="config/prompt.json", help='Prompt JSON file')
    parser.add_argument('--model_cfg', type=str, default="config/model.json", help='Model JSON file')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
    parser.add_argument('--stock_file', type=str, default="data/stock_history.csv", help='history stock data file path')
    parser.add_argument('--eps_dir', type=str, default="data/eps_history/", help='history eps data dir path')
    parser.add_argument('--ticker', type=str, default="AAPL", help='stock ticker')
    parser.add_argument('--start_time', type=str, default="2020-12-01", help='start time')
    parser.add_argument('--end_time', type=str, default="2021-01-01", help='end time')
    parser.add_argument('--narrative', action='store_true', default=False, help='use narrative input string')
    parser.add_argument('--image', action='store_true', default=False, help='use image input string')
    parser.add_argument('--quant', action='store_true', default=False, help='use quantization')
    
    return parser.parse_args()


def load_json(prompt_path, model_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(model_path, "r") as model_path:
        model_dict = json.load(model_path)

    return prompt_dict, model_dict


def load_stock_data(path):
    prices = pd.read_csv(path, header=[0, 1], index_col=0)
    return prices


class client(object):
    def __init__(self, url=None, headers=None, openai=False, model=None, pipe=None, gen_cfg=None, image_input=False):
        self.url = url
        self.headers = headers
        self.openai = openai
        self.client = OpenAI() if openai else None
        self.model = model
        self.pipe = pipe
        self.gen_cfg = gen_cfg
        self.image_input = image_input
    
    def query(self, message, image=None):
        # TODO: Add image input for API
        if self.openai:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": message}],
            )
            return completion.choices[0].message.content
        elif not self.pipe:
            payload = {"inputs": message}
            response = requests.post(self.url, headers=self.headers, json=payload)
            return response.json()[0]['generated_text']
        else:
            if self.image_input and image is not None:
                response = self.pipe(image, prompt=message, generate_kwargs=self.gen_cfg.to_dict())[0]['generated_text']
            else:
                response = self.pipe(message, generation_config=self.gen_cfg, clean_up_tokenization_spaces=True)[0]['generated_text']
            return response
 
            
def init_model(model_id, model_dict, api=False, token=None, image=False, quant=False):
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
    prompt_template = prompt_dict[model]['prompt']
    message = prompt_template.format(*instruction)
    return message


def construct_assistant_message():
    pass


def construct_instruction(args):
    question = "Stock trending prediction based off historical stocks' price and EPS data."
    background = "EPS (Earnings Per Share) is a widely used metric to gauge a company's profitability on a per-share basis. It's calculated as the company's net income divided by the number of outstanding shares. EPS Estimate refers to the projected (or expected) EPS for a company for a specific period, usually forecasted by financial analysts. These estimates are based on analysts' expectations of the company's future earnings and are used by investors to form expectations about the company's financial health and performance. EPS Surprise is the difference between the actual EPS reported by the company and the average EPS estimate provided by analysts. It's a key metric because it can significantly affect a stock's price. A positive surprise (actual EPS higher than expected) typically boosts the stock price, while a negative surprise (actual EPS lower than expected) usually causes the stock price to fall."
    criterion = "According to the historical stock price and EPS data, predict the stock trending after the lastest EPS report date with the EPS surprise reported."
    stock_s, stock_n = construct_stock_history(args.stock_file, args.ticker, args.start_time, args.end_time)
    eps_s, eps_n = construct_eps_history(args.eps_dir, args.ticker, args.start_time, args.end_time)
    if args.narrative:
        instruction = [question, background, criterion, stock_n, eps_n]
    else:
        instruction = [question, background, criterion, stock_s, eps_s]
    return instruction


def construct_stock_history(file, ticker, start, end):
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
    # Creating EPS signals
    offset = 10
    all_dates = pd.date_range(start, end, freq='D')
    low_prices = comp_stock['Low'].reindex(all_dates).interpolate(method='linear')
    high_prices = comp_stock['High'].reindex(all_dates).interpolate(method='linear')
    eps_signal = high_prices.loc[eps_markers] + offset
    fiscal_signal = low_prices.loc[fiscal_markers] - offset
    # Plotting the candlestick chart
    fig, axlist = mpf.plot(
        comp_stock, type='candle', mav=7, style='yahoo', 
        panel_ratios=(2,1), figratio=(2,1), figscale=1, 
        title='Stock Price Chart with EPS Dates', ylabel='Stock Price', 
        volume=True, show_nontrading=True, returnfig=True,
    )

    eps_x = [mdates.date2num(date) for date in eps_signal.index]
    eps_y = [eps_signal[i] for i in eps_signal.index]
    fiscal_x = [mdates.date2num(date) for date in fiscal_signal.index]
    fiscal_y = [fiscal_signal[i] for i in fiscal_signal.index]

    axlist[0].scatter(eps_x, eps_y, s=50, marker='v', color='orange', label='EPS Reported Date')
    axlist[0].scatter(fiscal_x, fiscal_y, s=50, marker='^', color='blue', label='Fiscal End Date')
    axlist[0].legend(frameon=False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)  # Important: move the buffer's start to the beginning after saving
    
    return buf


def main():
    args = args_parser()
    prompt_dict, model_dict = load_json(args.prompt_cfg, args.model_cfg)
    client = init_model(args.model, model_dict, args.api, args.token, args.image, args.quant)
    stock_df = load_stock_data(args.stock_file)
    args.stock_file = stock_df
    instruction = construct_instruction(args)
    instruction[-2] = 'refer to input image' if args.image else None
    message = construct_message(args.model, prompt_dict, instruction)
    image_buf = construct_images(args.stock_file, args.eps_dir, args.ticker, args.start_time, args.end_time) if args.image else None
    image = Image.open(image_buf) if args.image else None
    response = client.query(message, image)
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(response, open(f"{args.output_dir}/{args.model}_{args.ticker}_{args.start_time}_{args.end_time}.json", "w"))
    image.save(f"{args.output_dir}/{args.model}_{args.ticker}_{args.start_time}_{args.end_time}.png")
    image_buf.close() if args.image else None
    print(response)


if __name__ == "__main__":
    main()
