#!/bin/bash
#SBATCH --account=lerman_316
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --constraint=a100-80gb
#SBATCH --gpus-per-task=a100:2
#SBATCH --mem=0
#SBATCH --time=23:59:59



torchrun --nproc_per_node=2 --master_port=8004 train.py \
    --model_name_or_path /home1/kchen035/llama2/hf \
    --data_path train_sample_01_11.json \
    --bf16 True \
    --output_dir incas_tuned_model_v1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 0 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True



