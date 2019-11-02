#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/Dual-Encoder
python reproduce_critiquing.py --data_dir data/CDsVinyl/ --load_path critiquing_hyperparameters/CDsVinyl/hyper_parameters_s.csv --save_path CDsVinyl_fmap/s-e-cde-vae/part4.csv 
