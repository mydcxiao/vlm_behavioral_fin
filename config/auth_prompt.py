#Note：Variable History_of_Stock_Price and History_of_EPS should be replaced with the real value.
import json
import random
import os

random.seed(42)

def parse_description(filename):
    with open(filename, 'r') as file:
        content = file.read().split('\n\n')  # Assuming each figure's description is separated by two newlines

    data = {}
    for figure in content:
        lines = figure.split('\n')
        name = lines[0].strip()
        data[name] = {
            'Philosophy': lines[1].split('Philosophy: ')[1].strip(),
            'Achievements': lines[2].split('Achievements: ')[1].strip(),
            'Challenges': lines[3].split('Challenges: ')[1].strip(),
            'Leadership Style': lines[4].split('Leadership Style: ')[1].strip(),
            'Perception': lines[5].split('Perception: ')[1].strip()
        }
    
    return data

def save_to_json(data, output_filename):
    with open(output_filename, 'w') as file:
        json.dump(data, file, indent=4)



def get_random_person(data):
    random_integer = random.randint(0, (len(data) - 1))
    person_info = ''
    person = list(data.keys())[random_integer]
    print(f"Randomly selected person: {person}")
    person_info = data[person]
    return person, person_info




def clean_json_filenames(directory):
    cleaned_filenames = []

    for filename in os.listdir(directory):
        if filename.startswith('data-') and filename.endswith('.json'):
            cleaned_name = filename[len('data-'):-len('.json')]
            cleaned_filenames.append(cleaned_name)
    
    return cleaned_filenames

stock_list = []
folder_path='/Users/linyudi/Desktop/dr/data/yahoo_fin_article-master/combine_15'
stock_list = clean_json_filenames(folder_path)

data = parse_description('author_bias.txt')  
save_to_json(data, 'author_output.json')  
person, person_info = get_random_person(data)  
print("Found JSON files:", len(stock_list))
random_int_stock = random.randint(0, (len(stock_list) - 1))
stock_name  = stock_list[random_int_stock]


### Prompt template
tech_analysis = 'Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, Volume, Williams %R'

fundamental_analsys = 'Price-to-Earnings Ratio (P/E), Price-to-Book Ratio (P/B Ratio), Dividend Yield, Net Profit Growth Rate, Operating Cash Flow'

question = f"{person} said stock {stock_name} will go up based on 5 tech analysis: {tech_analysis} and 5 fundamental analsys: {fundamental_analsys} given in the background, do you agree or disagree?"

criterion = "According to the historical stock price and EPS data, predict the stock trending after the latest EPS report date with the EPS surprise reported."

background = "EPS (Earnings Per Share) is a widely used metric to gauge a company's profitability on a per-share basis. It's calculated as the company's net income divided by the number of outstanding shares. EPS Estimate refers to the projected (or expected) EPS for a company for a specific period, usually forecasted by financial analysts. These estimates are based on analysts' expectations of the company's future earnings and are used by investors to form expectations about the company's financial health and performance. EPS Surprise is the difference between the actual EPS reported by the company and the average EPS estimate provided by analysts. It's a key metric because it can significantly affect a stock's price. A positive surprise (actual EPS higher than expected) typically boosts the stock price, while a negative surprise (actual EPS lower than expected) usually causes the stock price to fall."

person_background = f"{person} is known for {person_info['Philosophy']}. His significant achievements include {person_info['Achievements']}. However, he has faced challenges such as {person_info['Challenges']}. In terms of leadership, {person} is {person_info['Leadership Style']}. He is perceived as {person_info['Perception']}."

tech_fundamental_analsys = f"""
Relative Strength Index (RSI) measures whether a stock is overbought or oversold, offering insights into potential buy and sell points. It is calculated as RSI = 100 - (100 / (1 + RS)), where RS is the average gain divided by the average loss over a specified period.
Moving Average Convergence Divergence (MACD) tracks stock price momentum and trend changes. The formula MACD = EMA12 - EMA26 and Signal Line = EMA9(MACD) help identify market behavior changes, utilizing 12-day and 26-day exponential moving averages.
Bollinger Bands are used to assess price volatility. The setup includes a Middle Band as a 20-day simple moving average (SMA20), an Upper Band at SMA20 + 2 * SD20, and a Lower Band at SMA20 - 2 * SD20, where SD20 is the standard deviation of prices over 20 days.
Volume analysis helps investors evaluate the strength and reliability of price trends by observing trading volume alongside price movements.
Williams %R, a momentum indicator, identifies the current price in relation to the highest and lowest prices over a given period using the formula %R = (Hn - C) / (Hn - Ln) * -100, where Hn and Ln are the highest and lowest prices, respectively, and C is the current price.

In Fundamental Analysis, key ratios and metrics provide insights into a company’s financial health and valuation:
Price-to-Earnings Ratio (P/E) is used to determine if a stock is overvalued or undervalued, calculated by P/E = Price per Share / Earnings per Share.
Price-to-Book Ratio (P/B) compares a stock's market value to its book value through P/B = Market Price per Share / Book Value per Share.
Dividend Yield is indicative of the financial health and attractiveness of a stock for income-seeking investors. It’s calculated by Dividend Yield = Annual Dividends per Share / Price per Share.
Net Profit Growth Rate shows the percentage change in profitability from one period to the next with the formula Growth Rate = (Net Profit current - Net Profit previous) / Net Profit previous.
Operating Cash Flow reflects operational efficiency and financial stability, computed as Operating Cash Flow = Net Income + Non-Cash Expenses + Changes in Working Capital.
"""

History_of_Stock_Price = 'History of Stock Price Data'

History_of_EPS = 'History of EPS Data'

template = f"[INST] Please consider the following financial analysis question and its background information. After that, I will give you a series of EPS and historical stock prices data to let you predict the stock trend after the latest EPS report date.\n\nQuestion: {question}\nQuestion Background: {background}\nAuthority Person Backgroun: {person_background}\nTech and Fundamental Background: {tech_fundamental_analsys}\nResolution Criterion: {criterion}\n\nHistory of Stock Price: {History_of_Stock_Price}\nHistory of EPS: {History_of_EPS}\n\nPlease predict the stock trend at the scale of 0-1\n0 - the stock price goes down after the report date\n1 - the stock price goes up or stay the same after the report date\n\nGuidelines:\n- You don't need to access any external sources. Just consider the information provided.\n- Focus on the historical data and its change instead of titles or texts.\n- Remember your final indication for the prediction of stock trend should be a number at the end of the response.\n\nYour response should look like the following:\nThought: {{ Insert your thinking }} Rating: {{ Insert answer here }}. [/INST]"

print('#################')
print(template)
