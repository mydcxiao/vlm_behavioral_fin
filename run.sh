export CUDA_VISIBLE_DEVICES=0
export BSZ=4
export TICKER=""
export MODEL_CFG="config/model.json"
export PROMPT_CFG="config/prompt.json"
export BIAS_TYPE="recency"
export NUM_SAMPLES=100
export SEED=42
export OPENAI_API_KEY=""
export ANTHROPIC_API_KEY=""
export API_KEY=""
export EPS_KEY=""
export TOKEN=""

export CUR_ID=6
MODELS=("llava" "MobileVLM" "MGM" "MiniCPM" "Phi-3-V" "gpt-4o" "claude-3-5" "gemini-1.5-pro")
MAX_NEW_TOKENS=(512 512 512 512 512 512 1024 2048)
TEMPERATURES=(0.0 0.0 0.2 0.7 0.0 1.0 0.0 0.0)
export MODEL=${MODELS[$CUR_ID]}
export MAX_NEW_TOKEN=${MAX_NEW_TOKENS[$CUR_ID]}
export TEMPERATURE=${TEMPERATURES[$CUR_ID]}

WINDOW_SIZES=(4 8 12 16 20)

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    # BIAS_DATA="./exp/${BIAS_TYPE}_bias/gpt-4o/exp_${BIAS_TYPE}_${WINDOW_SIZE}.json"
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
     --api \
    #  --bias_data $BIAS_DATA \
    #  --load_4bit \
    #  --token $TOKEN \
    #  --save_image \
    #  --load_8bit \
    #  --batch_inference \
    #  --collect_data \
    #  --collect_price \
    #  --collect_eps \
    #  --eps_key $EPS_KEY \
done
