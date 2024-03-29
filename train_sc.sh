#!/bin/bash
#SBATCH -J nogcn # ��ҵ���� test
#SBATCH -p defq # �ύ��Ĭ�ϵ� defq ����
#SBATCH -N 1 # ʹ�� 1 ���ڵ�
#SBATCH --ntasks-per-node=1 # ÿ���ڵ㿪�� 1 ������
#SBATCH --cpus-per-task=6 # ÿ������ռ�� 6 �� CPU ����
#SBATCH --gres=gpu:1 # ����� GPU ������Ҫ�ڴ��ж��� GPU ����,�˴�Ϊ 1
#SBATCH -t 2500 # �����������ʱ���� 100 ����
  source activate EPMEI
GPU_ID=0

dataset="14lap"
for seed in 41 42 43 44 45; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_sc.py  --model_type bertsub  \
    --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
    --data_dir data/ASTE-Data-V2-EMNLP2020_pro/14lap/  \
    --learning_rate 4e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 50  \
    --do_train --do_eval --evaluate_during_training --eval_all_checkpoints  --eval_logsoftmax\
    --seed $seed      \
    --test_file result/ner/14lap/train_ner43/ent_pred_test_pro.json \
    --use_ner_results \
    --use_typemarker --gcn --n_gcn 2\
    --output_dir result/re/$dataset/result$seed --overwrite_output_dir --focalloss
done;
# Average the scores
#python3 sumup.py scire scire-scibert
