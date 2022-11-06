alphas="0.0"  

for alpha in $alphas  
do  
echo $alpha  

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29519 conversationalGeneration.py \
# --learning_rate 1e-4 \
# --model_name_or_path 't5-base' \
# --output_dir ./ablation/large-1e-4-base-multi-et$alpha-multisetUnseen \
# --num_train_epochs 64 \
# --per_device_train_batch_size 8 \
# --per_device_eval_batch_size 8 \
# --warmup_ratio 0.10 \
# --fp16 false \
# --eval_steps 200 \
# --gradient_accumulation_steps 1 \
# --evaluation_strategy 'steps' \
# --logging_strategy 'steps' \
# --save_strategy 'steps' \
# --save_steps 200 \
# --logging_steps 200 \
# --train_file './data/multisetUnseen/t5_decision_information_roberta_base_train_all_snipped_id.json' \
# --validation_file './data/multisetUnseen/t5_decision_information_roberta_base_dev_all_snipped_id.json' \
# --test_file './data/multisetUnseen/t5_decision_information_roberta_base_test_all_snipped_id.json' \
# --max_source_length 512 \
# --max_target_length 256 \
# --pad_to_max_length false \
# --source_prefix "Conversational Machine Reading : " \
# --do_train true \
# --do_eval true \
# --do_predict true \
# --ddp_find_unused_parameters true \
# --overwrite_output_dir true \
# --prediction_loss_only false \
# --load_best_model_at_end true \
# --metric_for_best_model 'micro_accuracy'  \
# --predict_with_generate true \
# --greater_is_better true \
# --num_beams 10  \
# --encoder_classifier 1 \
# --loss_entailment $alpha \
# --loss_ce 0.0 \
# --decoder_enhance 0 \
# --classify_only  0 \
# --encoder_loss 1 \
# --save_total_limit 3 \
# > large-1e-4-base-multi-et$alpha-multisetUnseen.log 2>&1

seeds="9 110 123 3120 42 "  

for seed in $seeds 
do  
echo $seed  

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29519 conversationalGeneration.py \
--learning_rate 2e-4 \
--model_name_or_path 't5-base' \
--output_dir ./ablation/large-2e-4-base-multi-et$alpha-multisetUnseen-candidates-1-seed-$seed-noshuffle-long \
--num_train_epochs 6400 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--warmup_ratio 0.10 \
--fp16 false \
--eval_steps 200 \
--gradient_accumulation_steps 1 \
--evaluation_strategy 'steps' \
--logging_strategy 'steps' \
--save_strategy 'steps' \
--save_steps 200 \
--logging_steps 200 \
--train_file './data/multisetUnseen/t5_decision_information_roberta_base_train_all_snipped_id.json' \
--validation_file './data/multisetUnseen/t5_decision_information_roberta_base_dev_all_snipped_id.json' \
--test_file './data/multisetUnseen/t5_decision_information_roberta_base_test_all_snipped_id.json' \
--max_source_length 512 \
--max_target_length 256 \
--pad_to_max_length false \
--source_prefix "Conversational Machine Reading : " \
--do_train true \
--do_eval true \
--do_predict true \
--ddp_find_unused_parameters true \
--overwrite_output_dir true \
--prediction_loss_only false \
--load_best_model_at_end true \
--metric_for_best_model 'micro_accuracy'  \
--predict_with_generate true \
--greater_is_better true \
--num_beams 10  \
--encoder_classifier 1 \
--loss_entailment $alpha \
--loss_ce 0.0 \
--decoder_enhance 0 \
--classify_only  0 \
--encoder_loss 1 \
--save_total_limit 3 \
--seed $seed \
--rule_candidates 5 \
> large-2e-4-base-multi-et$alpha-multisetUnseen-candidates-1-seed-$seed-noshuffle-long.log 2>&1
done
done