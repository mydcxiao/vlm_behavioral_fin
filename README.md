# Recency Bias of Large Language Model in Predicting Market

## TODO:
- ✅ implement LLM inference
- ✅ implement multimodal inference
- ✅ build batched inference for massive data
- ✅ conduct pilot exp
- ✅ test prompt template
- ✅ implement other bias detection
- ✅ conduct full experiments

## Generation
```shell
run.sh
```
OR
```shell
python gen.py --model llava --image --quant --ticker AAPL AMZN
```

## Evaluation
```shell
python eval.py --path exp.json
```

## Prediction
LLM:
```shell
python predict.py --model llava --start_time "2020-12-01" --end_time "2021-01-01"
```

--------

Multimodal LLM:
```shell
python predict.py --model llava --image --quant --start_time "2020-01-01" --end_time "2021-01-01"
```
