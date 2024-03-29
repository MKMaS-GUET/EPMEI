#!/bin/bash
#SBATCH -J test2
#SBATCH -p defq 
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=6 
#SBATCH --gres=gpu:1 
#SBATCH -t 2500 
source activate EPMEI
GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_ner.py  --model_type bertspanmarker  \
    --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
    --data_dir data/ASTE-Data-V2-EMNLP2020_pro/14lap  \
    --learning_rate 5e-5  --num_train_epochs 10  --per_gpu_train_batch_size  4  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 50  --max_pair_length 128  --max_mention_ori_length 8    \
    --do_train --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --seed 43  --onedropout  --lminit  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir result/ner/14lap/train_ner43 --overwrite_output_dir  --output_results