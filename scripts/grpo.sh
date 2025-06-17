export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

model_name_or_path=Qwen/Qwen2.5-Coder-7B-Instruct
attn_implementation=flash_attention_2
output_dir=./results
lr=5e-5
lr_scheduler_type=cosine
logging_steps=10
max_steps=5000
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=1
max_prompt_length=2048
max_completion_length=1024
num_generations=8
grpo_beta=0.001
data_path=data/api_update.jsonl
lora_r=64
lora_alpha=64
lora_dropout=0.01
seed=42
save_steps=100
resume_from_checkpoint=None
warmup_ratio=0.03
reward_func=es_star
results=results

torchrun --nproc_per_node 2 --master-port 12345 src/GRPO/grpo.py \
    --model_name_or_path ${model_name_or_path} \
    --attn_implementation ${attn_implementation} \
    --output_dir ${output_dir} \
    --lr ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --logging_steps ${logging_steps} \
    --max_steps ${max_steps} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --bf16 \
    --max_prompt_length ${max_prompt_length} \
    --max_completion_length ${max_completion_length} \
    --num_generations ${num_generations} \
    --grpo_beta ${grpo_beta} \
    --data_path ${data_path} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --seed ${seed} \
    --save_steps ${save_steps} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --warmup_ratio ${warmup_ratio} \
    --reward_func ${reward_func} \
    --results ${results}

