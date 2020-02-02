#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForVAEBasedRecSys
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_rating_tuning --save_path CDsVinyl_final/CDsVinyl_final_result5.csv
