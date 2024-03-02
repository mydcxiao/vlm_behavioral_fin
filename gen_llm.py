import requests
import json
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, GemmaTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForPreTraining, AutoModel
from transformers import pipeline

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama", help='Model name')
    parser.add_argument('--api', action='store_true', default=False, help='Use API')
    parser.add_argument('--token', type=str, default="", help='API token')
    parser.add_argument('--prompt_cfg', type=str, default="config/prompt.json", help='Prompt JSON file')
    parser.add_argument('--model_cfg', type=str, default="config/model.json", help='Model JSON file')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
    
    return parser.parse_args()

def load_json(prompt_path, model_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(model_path, "r") as model_path:
        model_dict = json.load(model_path)

    return prompt_dict, model_dict

class client(object):
    def __init__(self, url=None, headers=None, openai=False, model=None, pipe=None):
        self.url = url
        self.headers = headers
        self.openai = openai
        self.client = OpenAI() if openai else None
        self.model = model
        self.pipe = pipe
    
    def query(self, message):
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
            return self.pipe(message)[0]['generated_text']
            

def init_model(model_id, model_dict, api=False, token=None):
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
            pipe = pipeline("text-generation", model=model_id)
            return client(pipe=pipe)
        except Exception as e:
            raise e
            
def construct_message(model, prompt_dict, instruction):
    prompt_template = prompt_dict[model]['prompt']
    message = prompt_template.format(inst=instruction)
    return message

# def construct_assistant_message():
#     pass

def main():
    args = args_parser()
    prompt_dict, model_dict = load_json(args.prompt_cfg, args.model_cfg)
    client = init_model(args.model, model_dict, args.api, args.token)
    instruction = "Give me the prediction of the trend of the stock price for the next 5 days." + \
        "Your final answer should be up or down, in the form of \\boxed{{answer}}, at the end of your response."
    message = construct_message(args.model, prompt_dict, instruction)
    response = client.query(message)
    print(response)

if __name__ == "__main__":
    main()

# processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-7b")
# model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")

# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
# model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

# prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=prompt, images=image, return_tensors="pt")

# # Generate
# generate_ids = model.generate(**inputs, max_length=30)
# processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
