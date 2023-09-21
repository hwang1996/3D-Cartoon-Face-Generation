EXP=disney
CONFIG=disney
GPUS=1
PORT=${PORT:-29574}

mkdir -p results/${EXP}
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    run.py \
    --launcher pytorch \
    --config configs/${CONFIG}.yml \
    2>&1 | tee results/${EXP}/log.txt
