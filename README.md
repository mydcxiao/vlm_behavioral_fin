# Behavioral Bias of Vision-Language Models: A Behavioral Finance View

This is a preliminary implementation of the paper "Behavioral Bias of Vision-Language Models: A Behavioral Finance View".

## Generation
Run experiments for recency bias (`--bias_type recency`) and authority bias (`--bias_type authoritative`).

Full experiments:
```shell
run.sh
```

Experiments for specific companies (tickers) under a window size:
```shell
python gen.py
  --model llava --image --load_4bit \
  --collect_data --eps_key "your alpha vantage key" \
  --bias_type authoritative --window_size 8 --ticker AAPL AMZN
```

## Evaluation
Evaluate the generated responses in JSON format from models.

```shell
python eval.py --path exp.json
```

## Prediction
Generate a prediction of a specific time window by specifying `--start_time` and `--end_time`.

LLM:
```shell
python predict.py --model llama-2-7b-chat --start_time "2020-12-01" --end_time "2021-01-01"
```

--------

Multimodal LLM:
```shell
python predict.py --model llava --image --load_4bit --start_time "2020-01-01" --end_time "2021-01-01"
```
