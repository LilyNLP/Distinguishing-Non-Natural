# example code to train the detector against TextFooler on MR

# 1st stage
# 1. get artificial samples：
python run_randomization.py \
  --complex_threshold 3000 \
  --ratio 0.3 \
  --syn_num 30 \
  --most_freq_num 30 \
  --simplify_version random_freq_v1 \
  --file_to_simplify data/MR/original_data/train.tsv \
  --output_path data/MR/randomized_data/train_randomized.tsv

# 2. form train set containing artificial and origianl samples：
python change_label_for_tsv.py \
--input_file data/MR/randomized_data/train_randomized.tsv \
--output_file data/MR/randomized_data/train_randomized.tsv \
--change_to 1

python change_label_for_tsv.py \
--input_file data/MR/original_data/train.tsv \
--output_file data/MR/randomized_data/train_original.tsv \
--change_to 0

python concatenate.py \
--input_file_1 data/MR/randomized_data/train_randomized.tsv \
--input_file_2 data/MR/randomized_data/train_original.tsv \
--output_file data/MR/randomized_data/train.tsv

# 3. form test set containing artificial and origianl samples：
python run_randomization.py \
  --complex_threshold 3000 \
  --ratio 0.3 \
  --syn_num 30 \
  --most_freq_num 30 \
  --simplify_version random_freq_v1 \
  --file_to_simplify data/MR/original_data/test.tsv \
  --output_path data/MR/randomized_data/test_randomized.tsv
  
python change_label_for_tsv.py \
--input_file data/MR/randomized_data/test_randomized.tsv \
--output_file data/MR/randomized_data/test_randomized.tsv \
--change_to 1

python change_label_for_tsv.py \
--input_file data/MR/original_data/test.tsv \
--output_file data/MR/randomized_data/test_original.tsv \
--change_to 0

python concatenate.py \
--input_file_1 data/MR/randomized_data/test_randomized.tsv \
--input_file_2 data/MR/randomized_data/test_original.tsv \
--output_file data/MR/randomized_data/test.tsv

# 4. train the detector to distinguish between original and artificial samples:
python run_detector_classification.py \
--task_name mr \
--max_seq_len 128 \
--do_train \
--do_eval \
--data_dir data/MR/randomized_data \
--output_dir experiments_detector/mr/randomization \
--model_name_or_path bert-base-uncased \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 5 \
--svd_reserve_size 0 \
--evaluation_strategy epoch \
--overwrite_output_dir


# 2nd stage
# 1. use TextFooler to attack MR and get adversarial samples：
python attack_classification_simplified.py \
  --synonym_num 30 \
  --simplify_version v2 \
  --simp_sim_threshold 3000 \
  --dataset_path data/MR/original_data/train \
  --target_model bert \
  --target_model_path experiments/MR/baseline/bs16_lr3_ep5_base \
  --counter_fitting_cos_sim_path cos_sim_counter_fitting.npy \
  --USE_cache_path TextFooler/USE \
  --output_dir attack_detection/MR/bert \
  --data_size 9662 
  
python get_pure_adversaries.py \
	--adversaries_path attack_detection/MR/bert/adversaries.txt \
	--output_path attack_detection/MR/bert/ \
	--times 1 \
	--change 0 \
	--txtortsv txt \
	--datasize 9662

# 2. form train set containing adversarial and original samples: 
python form_dataset_detector.py \
	--abnormal_file attack_detection/MR/bert/adversaries_for_detection.txt \
	--abnormal_file_type txt \
	--abnormal_number 6164 \
	--normal_file data/MR/original_data/train.tsv \
	--normal_file_type tsv \
	--normal_number 6164 \
	--output_file data_detection/mr/textfooler/train.tsv \
	--output_file_type tsv
	
# 3. form test set containing adversarial and original samples:
python form_dataset_detector.py \
	--abnormal_file data/MR/original_data/pure_adversaries.txt \
	--abnormal_file_type txt \
	--abnormal_number 500 \
	--normal_file data/MR/original_data/test.tsv \
	--normal_file_type tsv \
	--normal_number 500 \
	--output_file data_detection/mr/textfooler/test.tsv \
	--output_file_type tsv

# 4. train the detector to distinguish between original and adversarial samples:
python run_detector_classification.py \
--task_name mr \
--max_seq_len 128 \
--do_train \
--do_eval \
--data_dir data_detection/mr/textfooler \
--output_dir experiments_detector/mr/textfooler \
--model_name_or_path experiments_detector/mr/randomization \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 5 \
--svd_reserve_size 0 \
--evaluation_strategy epoch \
--overwrite_output_dir
