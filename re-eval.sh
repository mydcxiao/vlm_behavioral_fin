export CUDA_VISIBLE_DEVICES=0
export BSZ=4
export TICKER=""
export MODEL_CFG="config/model.json"
export PROMPT_CFG="config/prompt.json"
export SEED=42
export OPENAI_API_KEY=""
export ANTHROPIC_API_KEY=""
export EPS_KEY=""
export TOKEN=""
export JSON_PATH=""

export CUR_ID=5
MODELS=("llava" "MobileVLM" "MGM" "MiniCPM" "Phi-3-V" "gpt-4o")
MAX_NEW_TOKENS=(512 512 512 512 512 512)
TEMPERATURES=(0.0 0.0 0.2 0.7 0.0 1.0)
export MODEL=${MODELS[$CUR_ID]}
export MAX_NEW_TOKEN=${MAX_NEW_TOKENS[$CUR_ID]}
export TEMPERATURE=${TEMPERATURES[$CUR_ID]}

python re-eval.py \
 --model $MODEL \
 --path $JSON_PATH \
 --batch_size $BSZ \
 --model_cfg $MODEL_CFG \
 --prompt_cfg $PROMPT_CFG \
 --max_new_tokens $MAX_NEW_TOKEN \
 --temperature $TEMPERATURE \
 --seed $SEED \
 --save_image \
 --api \
 --load_4bit \
#  --collect_price \
#  --eps_key $EPS_KEY \
 # --image \
 # --token $TOKEN \
 # --load_8bit \
 # --batch_inference \
 # --collect_data \
 # --collect_eps \

