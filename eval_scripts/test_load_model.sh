CHECKPOINT_PATH=checkpoints/gpt2_345m # need checkpoint path


VOCAB_FILE=tokenizer_utils/gpt2-vocab.json
MERGE_FILE=tokenizer_utils/gpt2-merges.txt
#GPT_ARGS=&#60;same as those in <a href="#gpt-pretraining">GPT pretraining</a> above&#62;


GPT_ARGS=" \
    --num-layers 2 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --fp16 \
    --vocab-file $VOCAB_FILE"


MAX_OUTPUT_SEQUENCE_LENGTH=256
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=2
OUTPUT_FILE=samples.json

python tools/generate_samples_gpt.py \
    $GPT_ARGS \
    --load $CHECKPOINT_PATH \
    --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
    --temperature $TEMPERATURE \
    --genfile $OUTPUT_FILE \
    --num-samples $NUMBER_OF_SAMPLES \
    --top_p $TOP_P \
    --recompute