from evaluation.critiquing_performance import critiquing_evaluation
from utils.io import save_dataframe_csv, load_yaml
from utils.modelnames import critiquing_models
from utils.progress import WorkSplitter

import pandas as pd
import tensorflow as tf


def critiquing(train_set, keyphrase_train_set, item_keyphrase_train_set, params, num_users_sampled, load_path, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = pd.read_csv(table_path + load_path)

    dfs_fmap = []

    for index, row in df.iterrows():

        if row['model'] not in critiquing_models:
            continue

        algorithm = row['model']
        rank = row['rank']
        beta = row['beta']
        lamb_l2 = row['lambda_l2']
        lamb_keyphrase = row['lambda_keyphrase']
        lamb_latent = row['lambda_latent']
        lamb_rating = row['lambda_rating']
        learning_rate = row['learning_rate']
        epoch = row['epoch']
        corruption = row['corruption']
        optimizer = row['optimizer']

        format = "model: {}, rank: {}, beta: {}, lambda_l2: {}, lambda_keyphrase: {}, " \
                 "lambda_latent: {}, lambda_rating: {}, learning_rate: {}, " \
                 "epoch: {}, corruption: {}, optimizer: {}"
        progress.section(format.format(algorithm, rank, beta, lamb_l2, lamb_keyphrase, lamb_latent, lamb_rating, learning_rate, epoch, corruption, optimizer))

        progress.subsection("Training")

        model = critiquing_models[algorithm](matrix_train=train_set,
                                             epoch=epoch,
                                             lamb_l2=lamb_l2,
                                             lamb_keyphrase=lamb_keyphrase,
                                             lamb_latent=lamb_latent,
                                             lamb_rating=lamb_rating,
                                             beta=beta,
                                             learning_rate=learning_rate,
                                             rank=rank,
                                             corruption=corruption,
                                             optimizer=optimizer,
                                             matrix_train_keyphrase=keyphrase_train_set)

        num_users, num_items = train_set.shape
        df_fmap = critiquing_evaluation(train_set, keyphrase_train_set, item_keyphrase_train_set, model, algorithm, num_users, num_items, num_users_sampled, topk=[5, 10, 20])

        dfs_fmap.append(df_fmap)

        model.sess.close()
        tf.reset_default_graph()

    df_output_fmap = pd.concat(dfs_fmap)

    save_dataframe_csv(df_output_fmap, table_path, name=save_path+'_FMAP.csv')

