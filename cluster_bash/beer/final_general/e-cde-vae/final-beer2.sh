#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/Dual-Encoder
python reproduce_general_results.py --data_dir data/beer/ --tuning_result_path beer_rating_tuning --save_path beer_rating_final/beer_final_result2.csv
