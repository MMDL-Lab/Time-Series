export CUDA_VISIBLE_DEVICES=0

model_name=DualMamba

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --c_out 7 \
  --patch_len 12 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --train_epochs 40 \
  --batch_size 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 12 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --train_epochs 40 \
  --batch_size 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 12 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --train_epochs 40 \
  --batch_size 256 \
  --itr 1

 python -u run.py \
   --is_training 1 \
   --root_path ./dataset/ETT-small/ \
   --data_path ETTh1.csv \
   --model_id ETTh1_96_720 \
   --model $model_name \
   --data ETTh1 \
   --features M \
   --seq_len 96 \
   --pred_len 720 \
   --e_layers 3 \
   --enc_in 7 \
   --dec_in 7 \
   --c_out 7 \
   --patch_len 12 \
   --des 'Exp' \
   --d_model 128 \
   --d_ff 128 \
   --train_epochs 40 \
   --batch_size 128 \
   --itr 1