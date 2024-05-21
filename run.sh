# export MODEL="llava"
export MODEL="MobileVLM"
# export MODEL="MGM"
export WINDOW_SIZE=20
export TICKER=""

python gen.py \
 --model $MODEL \
 --window_size $WINDOW_SIZE \
 --ticker $TICKER \
 --image \
 --load_4bit \
#  --load_8bit \
#  --save_image \
