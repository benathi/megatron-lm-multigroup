CHECKPOINT_PATH=checkpoints/gpt2nodes_large

VOCAB_FILE=tokenizer_utils/gpt2-vocab.json
MERGE_FILE=tokenizer_utils/gpt2-merges.txt
DATA_PATH=/mnt/efs/people/benathi/megatron/data/meg-gpt2-oscar-en-10k_text_document

N_GPUS_PER_NODE=8
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
TP_SIZE=8
PP_SIZE=16


## new  - BenA
NLAYERS=2

TRAIN_SAMPLES=146484375
LR_DECAY_SAMPLES=126953125
LR_WARMUP_SAMPLES=183105
LR=6.0e-5
MIN_LR=6.0e-5


CLIP_GRAD=1.0

VOCAB_SIZE=50257
########

SAVE_INTERVAL=1000
ZERO_STAGE=1

config_json=ds_config.json

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


echo \"content of config . json\"

more $config_json

MASTER_PORT=1377

MASTER_ADDR=172.31.19.2

echo \"New master address ${MASTER_ADDR}\"

NODE_RANK=$1


PYTHONUNBUFFERED=1 python -u -m torch.distributed.run --nproc_per_node $N_GPUS_PER_NODE --nnodes 2 --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT  --max_restarts 0 --tee 3 \
       pretrain_gpt_pl.py \
       --distributed-backend nccl \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --bf16 \
       --partition-activations \
       --seed 42 \
       --embed-layernorm \
       --data-path $DATA_PATH \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 16 \
              --num-layers 48 \
              --hidden-size 12288 \
              --num-attention-heads 96 \
              --seq-length 2048 \
              --max-position-embeddings 2048 \
       --micro-batch-size 1 \
       --global-batch-size 1536 \
       --rampup-batch-size 16 16 5859375 \
       --train-samples $TRAIN_SAMPLES \
              --lr-decay-samples $LR_DECAY_SAMPLES \
              --lr-warmup-samples $LR_WARMUP_SAMPLES \
              --lr $LR \
       --min-lr $MIN_LR \
              --lr-decay-style cosine \
              --log-interval 10 \
              --eval-iters 40 \
              --eval-interval 1000 \
       --data-path ${DATASET} \
       --vocab-file tokenizer_utils/gpt2-vocab.json \
       --merge-file tokenizer_utils/gpt2-merges.txt \
       --exit-interval 10000 \
       --save-interval 1000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
              --split 98,2,0 \
              --clip-grad $CLIP_GRAD \
       --weight-decay 0.1 \
       --optimizer adam \
              --adam-eps 1e-8 \
              --adam-beta1 0.9 \
              --adam-beta2 0.95 \
       --init-method-std 0.006 \
       --tensorboard-dir ${CHECKPOINT_PATH}/tensorboard \
              --tensorboard-queue-size 5 \
              --log-timers-to-tensorboard \
              --log-batch-size-to-tensorboard \
              --log-validation-ppl-to-tensorboard \
              --kill-switch-path /tmp/kill-switch \
       --deepspeed \
              --deepspeed_config ${config_json} \
              --zero-stage ${ZERO_STAGE} \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing