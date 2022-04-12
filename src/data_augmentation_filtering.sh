# example code to do filterd data augmentation on MR

# 1. generate filtered augmented data on MR
python run_randomization_limited_reverse.py \
  --complex_threshold 3000 \
  --ratio 0.3 \
  --syn_num 50 \
  --most_freq_num 50 \
  --simplify_version random_freq_v1 \
  --file_to_simplify data/MR/original_data/train.tsv \
  --output_path data/MR/randomized_data/train_randomized_limited_03_50.tsv \
  --detector_model_path experiments_detector/mr/new
  
# 2. combine augmented data with original_data to form train set
cp data/MR/randomized_data/train_randomized_limited_03_50.tsv  data/MR/randomized_data/train_randomized_limited_03_50_reverse.tsv
  
python concatenate.py \
--input_file_1 data/MR/randomized_data/train_randomized_limited_03_50_reverse.tsv \
--input_file_2 data/MR/original_data/train.tsv \
--output_file data/MR/augmentation_limited_03_50_reverse/train.tsv

# 3. test set
cp data/MR/original_data/test.tsv  data/MR/augmentation_limited_03_50_reverse/test.tsv

# 4. train BERT / RoBERTa / ELECTRA
python run_classification.py \
--task_name mr \
--max_seq_len 128 \
--do_train \
--do_eval \
--data_dir data/MR/augmentation_limited_03_50_reverse \
--output_dir data_augmentation/MR/augmentation_limited_03_50_reverse \
--model_name_or_path bert-base-uncased \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 6 \
--svd_reserve_size 0 \
--evaluation_strategy epoch \
--overwrite_output_dir

python run_classification.py \
--task_name mr \
--max_seq_len 128 \
--do_train \
--do_eval \
--data_dir data/MR/augmentation_limited_03_50_reverse \
--output_dir data_augmentation/MR/augmentation_limited_reverse_roberta \
--model_name_or_path roberta-base \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 6 \
--svd_reserve_size 0 \
--evaluation_strategy epoch \
--overwrite_output_dir

python run_classification.py \
--task_name mr \
--max_seq_len 128 \
--do_train \
--do_eval \
--data_dir data/MR/augmentation_limited_03_50_reverse \
--output_dir data_augmentation/MR/augmentation_limited_reverse_electra \
--model_name_or_path google/electra-base-discriminator \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 6 \
--svd_reserve_size 0 \
--evaluation_strategy epoch \
--overwrite_output_dir


