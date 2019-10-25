#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/Dual-Encoder
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl --save_path CDsVinyl/CDsVinyl_final/CDsVinyl_final_result2.csv
