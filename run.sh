export CUDA_VISIBLE_DEVICES=0
export BSZ=4
export TICKER=""
export MODEL_CFG="config/model.json"
export PROMPT_CFG="config/prompt.json"
export BIAS_TYPE="recency"
export NUM_SAMPLES=100
export SEED=42
export OPENAI_API_KEY=""
# export WINDOW_SIZE=5

export CUR_ID=5
MODELS=("llava" "MobileVLM" "MGM" "MiniCPM" "Phi-3-V" "gpt-4o")
MAX_NEW_TOKENS=(512 512 512 512 512 512)
TEMPERATURES=(0.0 0.0 0.2 0.7 0.0 1.0)
export MODEL=${MODELS[$CUR_ID]}
export MAX_NEW_TOKEN=${MAX_NEW_TOKENS[$CUR_ID]}
export TEMPERATURE=${TEMPERATURES[$CUR_ID]}

WINDOW_SIZES=(4 8 12 16 20)

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    python gen.py \
     --model $MODEL \
     --window_size $WINDOW_SIZE \
     --ticker $TICKER \
     --batch_size $BSZ \
     --model_cfg $MODEL_CFG \
     --prompt_cfg $PROMPT_CFG \
     --bias_type $BIAS_TYPE \
     --max_new_tokens $MAX_NEW_TOKEN \
     --temperature $TEMPERATURE \
     --num_samples $NUM_SAMPLES \
     --seed $SEED \
     --image \
     --load_4bit \
     --api \
    #  --save_image \
    #  --load_8bit \
    #  --batch_inference \
done
