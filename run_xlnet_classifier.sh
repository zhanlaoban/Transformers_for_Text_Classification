export CUDA_VISIBLE_DEVICES=0,1,2,3
for((i=0;i<1;i++));  
do   

TASK_NAME="THUNews"

python run_classifier.py \
  --model_type=bert \
  --model_name_or_path ./pretrained_model/bert_base \
  --data_dir ./data/data_$i \
  --output_dir ./results/bert/bert$i \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=4.0 \
  --logging_steps=14923 \
  --save_steps=14923 \
  --overwrite_output_dir

done
# 每一个epoch保存一次
# 每一个epoch评估一次