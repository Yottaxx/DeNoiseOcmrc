#export CUDA_VISIBLE_DEVICES='4'
 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29515 run_ranker.py \
  --model_name_or_path roberta-base \
  --train_file ./data/rank/sharc_train.json \
  --validation_file ./data/rank/sharc_dev.json \
  --test_file ./data/rank/sharc_test.json \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 128 \
  --output_dir ./rankerOut/ \
  --overwrite_output_dir true \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --evaluation_strategy 'steps' \
  --eval_steps 200 \
  --save_steps 200 \
  --logging_steps 200 \
  --save_total_limit 3 \
  --greater_is_better true \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy 'steps' \
  --load_best_model_at_end true \
  --ddp_find_unused_parameters false \
  --metric_for_best_model 'accuracy' \
  --gamma 2 \
  --alpha 0.75 \
  --fp16 true \
  >ranker2075.log 2>&1 &


#export CUDA_VISIBLE_DEVICES='4'
 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29515 run_ranker.py \
  --model_name_or_path roberta-base \
  --train_file ./data/rank/sharc_train.json \
  --validation_file ./data/rank/sharc_dev.json \
  --test_file ./data/rank/sharc_test.json \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 8 \
  --output_dir ./rankerOut/ \
  --overwrite_output_dir true \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --evaluation_strategy 'steps' \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 500 \
  --save_total_limit 3 \
  --greater_is_better true \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy 'steps' \
  --load_best_model_at_end true \
  --ddp_find_unused_parameters false \
  --metric_for_best_model 'accuracy' \
  --gamma 1 \
  --alpha 0.5 \
  >ranker305.log 2>&1

  #export CUDA_VISIBLE_DEVICES='4'
 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29515 run_ranker.py \
  --model_name_or_path roberta-base \
  --train_file ./data/rank/sharc_train.json \
  --validation_file ./data/rank/sharc_dev.json \
  --test_file ./data/rank/sharc_test.json \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 16 \
  --output_dir ./rankerOut/ \
  --overwrite_output_dir true \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --evaluation_strategy 'steps' \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 500 \
  --save_total_limit 3 \
  --greater_is_better true \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy 'steps' \
  --load_best_model_at_end true \
  --ddp_find_unused_parameters false \
  --metric_for_best_model 'accuracy' \
  --gamma 3 \
  --alpha 0.75 \
  >ranker3075.log 2>&1

  #export CUDA_VISIBLE_DEVICES='4'
 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29515 run_ranker.py \
  --model_name_or_path roberta-base \
  --train_file ./data/rank/sharc_train.json \
  --validation_file ./data/rank/sharc_dev.json \
  --test_file ./data/rank/sharc_test.json \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 16 \
  --output_dir ./rankerOut/ \
  --overwrite_output_dir true \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --evaluation_strategy 'steps' \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 500 \
  --save_total_limit 3 \
  --greater_is_better true \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy 'steps' \
  --load_best_model_at_end true \
  --ddp_find_unused_parameters false \
  --metric_for_best_model 'accuracy' \
  --gamma 2 \
  --alpha 0.5 \
  >ranker205.log 2>&1






