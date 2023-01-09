CHECKPOINT_PATH=checkpoints/gpt2pl

VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt
#DATA_PATH=data/meg-gpt2-oscar-en-10k_text_document
DATA_PATH=/mnt/efs/people/benathi/megatron/data/meg-gpt2-oscar-en-10k_text_document
TENSORBOARD_PATH=output_dir/tensorboard

N_GPUS=2
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TP_SIZE=2
PP_SIZE=1

NLAYERS=2
NHIDDEN=8
NHEADS=2
SEQ_LEN=512
VOCAB_SIZE=50257
SAVE_INTERVAL=50
TRAIN_SAMPLES=10_000
ZERO_STAGE=1

config_json="./ds_config.json"

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


MASTER_ADDR=localhost
#MASTER_PORT=6777
MASTER_PORT=1377


PYTHONUNBUFFERED=1 python -u -m torch.distributed.launch \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --master_port 1337 \
    --max_restarts 0 \
    --tee 3 \
    pretrain_gpt_pl.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 2 2 1_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr-warmup-samples 5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 12 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --embed-layernorm \
    --fp16 \
    --partition-activations \
    --seed 42 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --exit-interval 10000 \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --kill-switch-path /tmp/kill-switch \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
