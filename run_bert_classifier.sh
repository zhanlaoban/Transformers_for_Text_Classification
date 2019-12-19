export CUDA_VISIBLE_DEVICES=0

TASK_NAME="THUNews"

python run_bert.py \
  --task_name=$TASK_NAME \
  --model_type=bert \
  --model_name_or_path ./pretrained_models/bert_base \
  --data_dir ./dataset/THUNews/5_5000 \
  --output_dir ./results/THUNews/5_5000 \
  --do_predict \
  --do_lower_case \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --logging_steps=14923 \
  --save_steps=14923 \
  --overwrite_output_dir

# 每一个epoch保存一次
# 每一个epoch评估一次