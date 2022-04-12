# example code to defend BERT against textfooler on MR
python textfooler_defense.py \
  --synonym_num 30 \
  --simplify_version v2 \
  --simp_sim_threshold 3000 \
  --dataset_path data/MR/original_data/test \
  --target_model bert \
  --target_model_path experiments/MR/baseline/bs16_lr3_ep5_base \
  --counter_fitting_cos_sim_path cos_sim_counter_fitting.npy \
  --USE_cache_path TextFooler/USE \
  --output_dir textfooler_defense/0 \
  --data_size 500 \
  --detector_path experiments_detector/mr/randomization