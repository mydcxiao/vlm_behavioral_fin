export MODEL="llava"
export WINDOW_SIZE=8
export TICKER=""

python gen.py \
 --model $MODEL \
 --window_size $WINDOW_SIZE \
 --ticker $TICKER \
 --image \
 --quant \
#  --save_image \