# example code to get the robustness under anomaly constraint on MR

# textfooler
python textfooler_limit.py \
  --synonym_num 30 \
  --simplify_version v2 \
  --simp_sim_threshold 3000 \
  --dataset_path data/MR/original_data/test \
  --target_model bert \
  --target_model_path experiments/MR/baseline/bs16_lr3_ep5_base \
  --detector_model_path experiments_detector/mr/textfooler \
  --counter_fitting_cos_sim_path cos_sim_counter_fitting.npy \
  --USE_cache_path TextFooler/USE \
  --output_dir textfooler_limit/ \
  --data_size 500

  
# bertattack
python bertattack_limit.py \
	--data_path data/MR/original_data/test.tsv \
	--mlm_path bert-base-uncased \
	--tgt_path experiments/MR/baseline/bs16_lr3_ep5_base \
	--detector_model_path experiments_detector/mr/bertattack \
	--use_sim_mat 1 \
	--output_dir bertattack/MR_adversarial_test.txt \
	--num_label 2 \
	--use_bpe 1 \
	--k 48 \
	--start 0 \
	--end 500 \
	--threshold_pred_score 0

	
#roberta
python textfooler_limit.py \
  --synonym_num 30 \
  --simplify_version v2 \
  --simp_sim_threshold 3000 \
  --dataset_path data/MR/original_data/test \
  --target_model bert \
  --target_model_path experiments/MR/roberta/baseline \
  --detector_model_path experiments_detector/mr/textfooler \
  --counter_fitting_cos_sim_path cos_sim_counter_fitting.npy \
  --USE_cache_path TextFooler/USE \
  --output_dir textfooler_limit/ \
  --data_size 500
  
# electra
python textfooler_limit.py \
  --synonym_num 30 \
  --simplify_version v2 \
  --simp_sim_threshold 3000 \
  --dataset_path data/MR/original_data/test \
  --target_model bert \
  --target_model_path experiments/MR/electra/baseline \
  --detector_model_path experiments_detector/mr/textfooler \
  --counter_fitting_cos_sim_path cos_sim_counter_fitting.npy \
  --USE_cache_path TextFooler/USE \
  --output_dir textfooler_limit/ \
  --data_size 500
  



