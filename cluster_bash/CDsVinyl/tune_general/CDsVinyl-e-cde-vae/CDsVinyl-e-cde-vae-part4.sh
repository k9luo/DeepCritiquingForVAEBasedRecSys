#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/Dual-Encoder
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl_rating_tuning/e_cdevae_tuning_part4.csv --parameters config/CDsVinyl/e-cde-vae-tune-rating/e-cde-vae-part4.yml
