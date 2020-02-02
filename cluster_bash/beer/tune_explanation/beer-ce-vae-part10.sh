#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForVAEBasedRecSys
python tune_parameters.py --data_dir data/beer/ --save_path beer_explanation_tuning/ce_vae_tuning_part10.csv --parameters config/beer/ce-vae-tune-keyphrase/ce-vae-part10.yml --tune_explanation
