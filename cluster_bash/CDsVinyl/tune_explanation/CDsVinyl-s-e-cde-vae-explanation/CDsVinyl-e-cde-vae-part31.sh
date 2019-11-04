#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/Dual-Encoder
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning_part31.csv --parameters config/CDsVinyl/s-e-cde-vae-tune-keyphrase/s-e-cde-vae-part31.yml --tune_explanation