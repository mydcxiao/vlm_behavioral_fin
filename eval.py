import requests
import json
import time
import argparse
import os
import io
import re

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to generated JSON file')
    parser.add_argument('--prompt_cfg', type=str, default="config/prompt.json", help='Prompt JSON file')
    parser.add_argument('--model', type=str, default="llama-chat", help='Model name')
    
    return parser.parse_args()


def parse_answer(response, split, pattern):
    answer = response.split(split)[-1]
    parts = pattern.findall(answer)
    
    try:
        number = float(parts[-1])
        return 1 if number >= 0.5 else 0
    except:
        return None
    
if __name__ == '__main__':
    args = args_parser()
    
    eval_list = json.load(open(args.path, 'r'))
    prompt_dict = json.load(open(args.prompt_cfg, 'r'))
    split = prompt_dict[args.model]['split'] if 'split' in prompt_dict[args.model] else None
    pattern = re.compile(r"(?:\{)?(\d+\.\d*|\d+|\.\d+)(?:\})?")
    
    correct, wrong, wrong_by_bias, no_answer, total = 0, 0, 0, 0, 0
    for eval in eval_list:
        answer = parse_answer(eval['response'], split, pattern)
        bias = eval['bias']
        gt = eval['gt']
        if answer == gt:
            correct += 1
        else:
            wrong += 1
            if bias == answer:
                wrong_by_bias += 1
            if bias is None:
                no_answer += 1
        total += 1
    
    stats = f"accuracy: {correct/total if total else 0}\nwrong by bias: {wrong_by_bias}\nbias percentage: {wrong_by_bias/total if total else 0}\nwrong by bias percentage: {wrong_by_bias/wrong if wrong else 0}\nno answer: {no_answer}\nno answer percentage: {no_answer/total if total else 0}"
    
    with open(args.path.replace('.json', '.txt'), "w") as f:
        f.write(stats)
        
    print(stats)