from tqdm import tqdm

import numpy as np


def predict(prediction_score, topK, matrix_Train):

    prediction = []

    for user_index in tqdm(range(prediction_score.shape[0])):
        vector_prediction = prediction_score[user_index]
        vector_train = matrix_Train[user_index]
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(vector_prediction, vector_train, topK=topK)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return np.vstack(prediction)

def sub_routine(vector_prediction, vector_train, topK=500):

    train_index = vector_train.nonzero()[1]

    vector_predict = vector_prediction

    candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
    vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]

