export CUDA_VISIBLE_DEVICES=0

TASK_NAME="THUNews"

python run.py \
  --task_name=$TASK_NAME \
  --model_type=xlnet \
  --model_name_or_path ./pretrained_models/xlnet_mid \
  --data_dir ./dataset/THUNews/5_5000 \
  --output_dir ./results/THUNews/xlnet_mid \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --max_seq_length=512 \
  --per_gpu_train_batch_size=2 \
  --per_gpu_eval_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --logging_steps=14923 \
  --save_steps=14923 \
  --overwrite_output_dir \
  --filter_sizes='3,4,5' \
  --filter_num=256 \
  --lstm_layers=1 \
  --lstm_hidden_size=512 \
  --lstm_dropout=0.1 \
  --gru_layers=1 \
  --gru_hidden_size=512 \
  --gru_dropout=0.1 \



# 每一个epoch保存一次
# 每一个epoch评估一次