#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForVAEBasedRecSys
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl_rating_tuning/ce_vae_tuning_part13.csv --parameters config/CDsVinyl/ce-vae-tune-rating/ce-vae-part13.yml
