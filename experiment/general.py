from evaluation.general_performance import evaluate
from prediction.predictor import predict
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml, find_best_hyperparameters
from utils.modelnames import models
from utils.progress import WorkSplitter

import pandas as pd
import tensorflow as tf


def general(train, test, params, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = find_best_hyperparameters(table_path + params['tuning_result_path'], 'NDCG')
#    import ipdb; ipdb.set_trace()

    try:
        output_df = load_dataframe_csv(table_path, save_path)
    except:
        output_df = pd.DataFrame(columns=['model', 'rank', 'lambda', 'topK', 'learning_rate', 'epoch', 'corruption', 'optimizer'])

    for index, row in df.iterrows():

        algorithm = row['model']
        rank = row['rank']
        lamb = row['lambda']
        learning_rate = row['learning_rate']
        epoch = 10
        corruption = row['corruption']
        optimizer = row['optimizer']

        row['topK'] = [5, 10, 15, 20, 50]
        row['metric'] = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        format = "model: {0}, rank: {1}, lambda: {2}, learning_rate: {3}, " \
                 "epoch: {4}, corruption: {5}, optimizer: {6}"

        progress.section(format.format(algorithm, rank, lamb, learning_rate, epoch, corruption, optimizer))

        progress.subsection("Training")

        model = models[algorithm](matrix_train=train,
                                  epoch=epoch,
                                  lamb=lamb,
                                  learning_rate=learning_rate,
                                  rank=rank,
                                  corruption=corruption,
                                  optimizer=optimizer)

        progress.subsection("Prediction")

        prediction_score = model.inference(train.todense())
        prediction = predict(prediction_score,
                             topK=row['topK'][-1],
                             matrix_Train=train)

        progress.subsection("Evaluation")

        result = evaluate(prediction, test, row['metric'], row['topK'])

        result_dict = {'model': algorithm,
                       'rank': rank,
                       'lambda': lamb,
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
