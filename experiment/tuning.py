from evaluation.general_performance import evaluate
from prediction.predictor import predict
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
from utils.modelnames import models
from utils.progress import WorkSplitter

import pandas as pd
import tensorflow as tf


def hyper_parameter_tuning(train, validation, params, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']

    try:
        df = load_dataframe_csv(table_path, save_path)
    except:
        df = pd.DataFrame(columns=['model', 'rank', 'lambda', 'topK', 'learning_rate', 'epoch', 'corruption', 'optimizer'])

    for algorithm in params['models']:

        for rank in params['rank']:

            for lamb in params['lambda']:

                for learning_rate in params['learning_rate']:

                    for epoch in params['epoch']:

                        for corruption in params['corruption']:

                            for optimizer in params['optimizer']:

                                if ((df['model'] == algorithm) &
                                    (df['rank'] == rank) &
                                    (df['lambda'] == lamb) &
                                    (df['learning_rate'] == learning_rate) &
                                    (df['epoch'] == epoch) &
                                    (df['corruption'] == corruption) &
                                    (df['optimizer'] == optimizer)).any():
                                    continue

                                format = "model: {0}, rank: {1}, lambda: {2}, " \
                                         "learning_rate: {3}, epoch: {4}, " \
                                         "corruption: {5}, optimizer: {6}"
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
#                                import ipdb; ipdb.set_trace()

                                prediction_score = model.inference(train.todense())
                                prediction = predict(prediction_score,
                                                     topK=params['topK'][-1],
                                                     matrix_Train=train)

                                progress.subsection("Evaluation")

                                result = evaluate(prediction, validation, params['metric'], params['topK'])

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

                                df = df.append(result_dict, ignore_index=True)

                                model.sess.close()
                                tf.reset_default_graph()

                                save_dataframe_csv(df, table_path, save_path)

