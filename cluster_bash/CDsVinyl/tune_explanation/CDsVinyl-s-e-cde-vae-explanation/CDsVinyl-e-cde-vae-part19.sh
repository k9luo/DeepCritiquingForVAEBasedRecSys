#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/Dual-Encoder
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning_part19.csv --parameters config/CDsVinyl/s-e-cde-vae-tune-keyphrase/s-e-cde-vae-part19.yml --tune_explanation