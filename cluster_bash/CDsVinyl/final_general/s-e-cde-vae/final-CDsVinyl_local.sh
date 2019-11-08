#!/usr/bin/env bash

python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_rating_final/s-e-cde-vae/CDsVinyl_final_result1.csv
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_rating_final/s-e-cde-vae/CDsVinyl_final_result2.csv
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_rating_final/s-e-cde-vae/CDsVinyl_final_result3.csv
#python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_rating_final/s-e-cde-vae/CDsVinyl_final_result4.csv
#python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_rating_final/s-e-cde-vae/CDsVinyl_final_result5.csv

python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_explanation_final/s-e-cde-vae/CDsVinyl_final_result1.csv --final_explanation
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_explanation_final/s-e-cde-vae/CDsVinyl_final_result2.csv --final_explanation
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_explanation_final/s-e-cde-vae/CDsVinyl_final_result3.csv --final_explanation
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_explanation_final/s-e-cde-vae/CDsVinyl_final_result4.csv --final_explanation
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning --save_path CDsVinyl_explanation_final/s-e-cde-vae/CDsVinyl_final_result5.csv --final_explanation
