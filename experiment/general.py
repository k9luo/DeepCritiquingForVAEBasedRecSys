from evaluation.general_performance import evaluate
from prediction.predictor import predict
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml, find_best_hyperparameters
from utils.modelnames import models
from utils.progress import WorkSplitter

import pandas as pd
import tensorflow as tf


def general(train, test, keyphrase_train, keyphrase_test, params, save_path, final_explanation=False):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = find_best_hyperparameters(table_path + params['tuning_result_path'], 'NDCG')

    try:
        output_df = load_dataframe_csv(table_path, save_path)
    except:
        output_df = pd.DataFrame(columns=['model', 'rank', 'beta', 'lambda_l2', 'lambda_keyphrase', 'lambda_latent', 'lambda_rating', 'topK', 'learning_rate', 'epoch', 'corruption', 'optimizer'])

    for index, row in df.iterrows():

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

        row['topK'] = [5, 10, 15, 20, 50]
        row['metric'] = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        format = "model: {}, rank: {}, beta: {}, lambda_l2: {}, lambda_keyphrase: {}, " \
                 "lambda_latent: {}, lambda_rating: {}, learning_rate: {}, " \
                 "epoch: {}, corruption: {}, optimizer: {}"

        progress.section(format.format(algorithm, rank, beta, lamb_l2, lamb_keyphrase, lamb_latent, lamb_rating, learning_rate, epoch, corruption, optimizer))

        progress.subsection("Training")

        model = models[algorithm](matrix_train=train,
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
                                  matrix_train_keyphrase=keyphrase_train)

        progress.subsection("Prediction")

        rating_score, keyphrase_score = model.predict(train.todense())

        progress.subsection("Evaluation")

        if final_explanation:
            prediction = predict_keyphrase(keyphrase_score,
                                           topK=row['topK'][-1])

            result = evaluate(prediction,
                              keyphrase_test,
                              row['metric'],
                              row['topK'])
        else:
            prediction = predict(rating_score,
                                 topK=row['topK'][-1],
                                 matrix_Train=train)


            result = evaluate(prediction, test, row['metric'], row['topK'])

        result_dict = {'model': algorithm,
                       'rank': rank,
                       'beta': beta,
                       'lambda_l2': lamb_l2,
                       'lambda_keyphrase': lamb_keyphrase,
                       'lambda_latent': lamb_latent,
                       'lambda_rating': lamb_rating,
                       'learning_rate': learning_rate,
                       'epoch': epoch,
                       'corruption': corruption,
                       'optimizer': optimizer}

        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4),
                                 round(result[name][1], 4)]

        output_df = output_df.append(result_dict, ignore_index=True)

        model.sess.close()
        tf.reset_default_graph()

        save_dataframe_csv(output_df, table_path, save_path)

    return output_df

