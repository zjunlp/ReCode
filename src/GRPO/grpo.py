import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer  # RL lib
from peft import get_peft_model, LoraConfig

from data import APIUpdateDataset
from utils import seed_torch
from reward import format_reward, em_star, es_star


def main(args):
    seed_torch(args.seed)
    
    # prepare data
    dataset = APIUpdateDataset(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset.tokenize(tokenizer)
    
    # we use Qwen2.5-Coder-7B as our base model
    model_name_or_path = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        # attn_implementation=args.attn_implementation,
        cache_dir='/mnt/16t/haoze',
    )
    dora_config = LoraConfig(
        use_dora=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model_name_or_path = get_peft_model(model_name_or_path, dora_config)
    
    # config for RL training
    training_args = GRPOConfig(
        output_dir=args.results,
        logging_dir=f'logs/{args.model_name_or_path}',
        report_to=['tensorboard'],
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        save_strategy="steps",
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        use_vllm=False,
        vllm_gpu_memory_utilization=1.0,
        # GRPO specific parameters
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.grpo_beta,
    )
    
    tokenizer.padding_side  = 'left'

    # setup trainer
    trainer = GRPOTrainer(
        model=model_name_or_path,
        reward_funcs=[format_reward, {"em_star": em_star, "es_star": es_star}[args.reward_func]],
        args=training_args,
        train_dataset=dataset,
        peft_config=dora_config,
        processing_class=tokenizer,
    )
    
    # train
    trainer.train()
    
    # Save model
    trainer.save_model(training_args.output_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--grpo_beta", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default="data/api_update.jsonl")
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--reward_func", type=str, default="es_star", choices=["es_star", "em_star"])
    
    main(parser.parse_args())
