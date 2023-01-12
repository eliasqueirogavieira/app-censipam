#!bin/bash

curr_dir=$(dirname $0)

~/conda/envs/censipam/bin/python main_prediction.py \
	--data_path /censipam_data/Datasets/sentinel_data/Sentinel_1A_10_615_21Mai2022_26Jun2022 \
 	--root_output /censipam_data/renam/tmp_SENTINEL_tmp \
	--config configs/cfg_sentinel.yml &
