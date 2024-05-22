export CUDA_VISIBLE_DEVICES=1
# export MODEL="llava"
# export MODEL="MobileVLM"
export MODEL="MGM"
export WINDOW_SIZE=16
export BSZ=1
export TICKER=""
export MODEL_CFG="config/model.json"
export PROMPT_CFG="config/prompt.json"

python gen.py \
 --model $MODEL \
 --window_size $WINDOW_SIZE \
 --ticker $TICKER \
 --batch_size $BSZ \
 --model_cfg $MODEL_CFG \
 --prompt_cfg $PROMPT_CFG \
 --image \
 --load_4bit \
#  --load_8bit \
#  --save_image \
