# Recency Bias of Large Language Model in Predicting Market

## TODO:
- ✅ implement LLM inference
- ✅ implement multimodal inference
- ✅ build batched inference for massive data
- :x: test prompt template
- :x: implement other bias detection
- :x: conduct experiments

## Evaluation
```shell
python eval.py --model llama --image --quant --ticker AAPL AMZN
```

## Prediction
LLM:
```shell
python predict.py --model llama --start_time "2020-12-01" --end_time "2021-01-01"
```

--------

Multimodal LLM:
```shell
python predict.py --model llava --image --quant --start_time "2020-01-01" --end_time "2021-01-01"
```
