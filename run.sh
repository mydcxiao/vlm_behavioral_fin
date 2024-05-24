export CUDA_VISIBLE_DEVICES=0
export BSZ=4
export TICKER=""
export MODEL_CFG="config/model.json"
export PROMPT_CFG="config/prompt.json"
export BIAS_TYPE="recency"
# export WINDOW_SIZE=5

MODELS=("llava" "MobileVLM" "MGM")
export MODEL=${MODELS[2]}

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
     --image \
     --load_4bit \
    #  --save_image \
    #  --load_8bit \
    #  --batch_inference \
done
