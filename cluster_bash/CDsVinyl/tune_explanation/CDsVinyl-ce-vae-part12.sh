#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForVAEBasedRecSys
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl_explanation_tuning/ce_vae_tuning_part12.csv --parameters config/CDsVinyl/ce-vae-tune-keyphrase/ce-vae-part12.yml --tune_explanation
