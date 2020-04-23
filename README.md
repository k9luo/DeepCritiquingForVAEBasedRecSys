Deep Critiquing for VAE-based Recommender Systems
====================================================================
![](https://img.shields.io/badge/linux-ubuntu-red.svg)
![](https://img.shields.io/badge/Mac-OS-red.svg)

![](https://img.shields.io/badge/cuda-10.0-green.svg)
![](https://img.shields.io/badge/python-2.7-green.svg)
![](https://img.shields.io/badge/python-3.6-green.svg)

![](https://img.shields.io/badge/cython-0.29-blue.svg)
![](https://img.shields.io/badge/fbpca-1.0-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.0.0-blue.svg)
![](https://img.shields.io/badge/numpy-1.15.2-blue.svg)
![](https://img.shields.io/badge/pandas-0.23.3-blue.svg)
![](https://img.shields.io/badge/pyyaml-4.1-blue.svg)
![](https://img.shields.io/badge/scipy-1.1.0-blue.svg)
![](https://img.shields.io/badge/seaborn-0.9.0-blue.svg)
![](https://img.shields.io/badge/sklearn-0.20.1-blue.svg)
![](https://img.shields.io/badge/tensorflow-1.12.0-blue.svg)
![](https://img.shields.io/badge/tqdm-4.28.1-blue.svg)


If you are interested in building up your research on this work, please cite:
```
@inproceedings{sigir20,
  author    = {Kai Luo and Hojin Yang and Ga Wu and Scott Sanner},
  title     = {Deep Critiquing for VAE-based Recommender Systems},
  booktitle = {Proceedings of the 43rd International {ACM} SIGIR Conference on Research and Development in Information Retrieval {(SIGIR-20)}},
  address   = {Xi'an, China},
  year      = {2020}
}
```

# Author Affiliate
<p align="center">
<a href="https://www.utoronto.ca//"><img src="https://github.com/k9luo/DeepCritiquingForVAEBasedRecSys/blob/master/logos/U-of-T-logo.svg" height="80"></a> | 
<a href="https://vectorinstitute.ai/"><img src="https://github.com/k9luo/DeepCritiquingForVAEBasedRecSys/blob/master/logos/vectorlogo.svg" height="80"></a> | 
</p>

# Algorithm Implemented
1. Critiquable and Explainable Variational Autoencoder (CE-VAE)

# Dataset
1. Amazon CDs&Vinyl,
2. Beer Advocate,

We don't have rights to release the datasets. Please ask permission from Professor [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/).

Please refer to the [`preprocess` folder](https://github.com/wuga214/DeepCritiquingForRecSys/tree/master/preprocess) for preprocessing raw datasets steps.

# Keyphrase
Keyphrases we used are not necessarily the best. If you are interested in how we extracted those keyphrases, please refer to the [`preprocess` folder](https://github.com/wuga214/DeepCritiquingForRecSys/tree/master/preprocess). If you are interested in what keyphrases we extracted, please refer to the [`data` folder](https://github.com/wuga214/DeepCritiquingForRecSys/tree/master/data).

# Example Commands

### General Recommendation and Explanation Single Run
```
python main.py --model CE-VAE --data_dir data/beer/ --epoch 300 --rank 100 --beta 0.001 --lambda_l2 0.0001 --lambda_keyphrase 0.01 --lambda_latent 0.01 --lambda_rating 1.0 --learning_rate 0.0001 --corruption 0.4 --topk 10 --disable_validation
```

Please check out the `cluster_bash` and `local_bash` folders for all commands details. Below are only example commands.

### General Recommendation Hyper-parameter Tuning
```
python tune_parameters.py --data_dir data/beer/ --save_path beer_rating_tuning/ce_vae_tuning_part1.csv --parameters config/beer/ce-vae-tune-rating/ce-vae-part1.yml
```

### Reproduce Final General Recommendation Performance
```
python reproduce_general_results.py --data_dir data/beer/ --tuning_result_path beer_rating_tuning --save_path beer_rating_final/beer_final_result1.csv
```

### Explanation Prediction Performance Hyper-parameter Tuning
```
python tune_parameters.py --data_dir data/beer/ --save_path beer_explanation_tuning/ce_vae_tuning_part1.csv --parameters config/beer/ce-vae-tune-keyphrase/ce-vae-part1.yml --tune_explanation
```

### Reproduce Final Explanation Prediction Performance
```
python reproduce_general_results.py --data_dir data/beer/ --tuning_result_path beer_explanation_tuning --save_path beer_explanation_final/beer_final_explanation_result1.csv --final_explanation
```

### Reproduce Critiquing
Find the hyperparameter set that is good for both rating and keyphrase prediction from tuning results for each dataset and put them in folder `tables/critiquing_hyperparameters/beer/hyper_parameters.csv` and `tables/critiquing_hyperparameters/CDsVinyl/hyper_parameters.csv`. Then run the following command.
```
python reproduce_critiquing.py --data_dir data/beer/ --model_saved_path beer --load_path explanation/beer/hyper_parameters.csv --num_users_sampled 1000 --save_path beer_fmap/beer_Critiquing
```

### Note
For baselines we used, please refer to [Noise Contrastive Estimation Projected Linear Recommender(NCE-PLRec)](https://github.com/wuga214/NCE_Projected_LRec).
