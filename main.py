from evaluation.general_performance import evaluate
from prediction.predictor import predict
from prediction.predictor import predict_keyphrase
from scipy import sparse
from utils.argcheck import check_float_positive, check_int_positive
from utils.io import load_numpy
from utils.modelnames import models
from utils.progress import inhour, WorkSplitter

import argparse
import numpy as np
import tensorflow as tf
import time


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyperparameter settings
    progress.section("Parameter Setting")

    print("Data Directory: {}".format(args.data_dir))
    print("Algorithm: {}".format(args.model))
    print("Optimizer: {}".format(args.optimizer))
    print("Corruption Rate: {}".format(args.corruption))
    print("Learning Rate: {}".format(args.learning_rate))
    print("Epoch: {}".format(args.epoch))
    print("Lambda L2: {}".format(args.lamb_l2))
    print("Lambda Keyphrase: {}".format(args.lamb_keyphrase))
    print("Lambda Latent: {}".format(args.lamb_latent))
    print("Lambda Rating: {}".format(args.lamb_rating))
    print("Beta: {}".format(args.beta))
    print("Rank: {}".format(args.rank))
    print("Train Batch Size: {}".format(args.train_batch_size))
    print("Predict Batch Size: {}".format(args.predict_batch_size))
    print("Evaluation Ranking Topk: {}".format(args.topk))
    print("Validation Enabled: {}".format(args.enable_validation))

    # Load Data
    progress.section("Load Data")
    start_time = time.time()

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    print("Train U-I Dimensions: {}".format(R_train.shape))

    R_train_keyphrase = load_numpy(path=args.data_dir, name=args.train_keyphrase_set).toarray()
    print("Train Keyphrase U-S Dimensions: {}".format(R_train_keyphrase.shape))

    if args.enable_validation:
        R_valid = load_numpy(path=args.data_dir, name=args.valid_set)
        R_valid_keyphrase = load_numpy(path=args.data_dir, name=args.valid_keyphrase_set)
    else:
        R_valid = load_numpy(path=args.data_dir, name=args.test_set)
        R_valid_keyphrase = load_numpy(path=args.data_dir, name=args.test_keyphrase_set)
    print("Elapsed: {}".format(inhour(time.time() - start_time)))

    progress.section("Preprocess Keyphrase Frequency")
    start_time = time.time()

    R_train_keyphrase[R_train_keyphrase != 0] = 1
    R_valid_keyphrase[R_valid_keyphrase != 0] = 1
    print("Elapsed: {}".format(inhour(time.time() - start_time)))

    progress.section("Train")
    start_time = time.time()

    model = models[args.model](matrix_train=R_train, epoch=args.epoch, lamb_l2=args.lamb_l2,
                               lamb_keyphrase=args.lamb_keyphrase, lamb_latent=args.lamb_latent,
                               lamb_rating=args.lamb_rating, beta=args.beta,
                               learning_rate=args.learning_rate, rank=args.rank,
                               corruption=args.corruption, optimizer=args.optimizer,
                               matrix_train_keyphrase=R_train_keyphrase)
    print("Elapsed: {}".format(inhour(time.time() - start_time)))

    progress.section("Predict")
    start_time = time.time()

    rating_score, keyphrase_score = model.predict(R_train.todense())
    prediction = predict(rating_score, args.topk, matrix_Train=R_train)
    print("Elapsed: {}".format(inhour(time.time() - start_time)))

    if args.enable_evaluation:
        progress.section("Create Metrics")
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
        result = evaluate(prediction, R_valid, metric_names, [args.topk])

        print("-")
        for metric in result.keys():
            print("{}:{}".format(metric, result[metric]))

        if keyphrase_score is not None:
            keyphrase_prediction = predict_keyphrase(keyphrase_score, args.topk)
            keyphrase_result = evaluate(keyphrase_prediction,
                                        sparse.csr_matrix(R_valid_keyphrase),
                                        metric_names,
                                        [args.topk])

            print("-")
            for metric in keyphrase_result.keys():
                print("{}:{}".format(metric, keyphrase_result[metric]))

        print("Elapsed: {}".format(inhour(time.time() - start_time)))

    model.sess.close()
    tf.reset_default_graph()


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="CE-VAE")

    parser.add_argument('--beta', dest='beta', default=0.2,
                        type=check_float_positive,
                        help='KL strength used in models. (default: %(default)s)')

    parser.add_argument('--corruption', dest='corruption', default=0.5,
                        type=check_float_positive,
                        help='Corruption rate used in models. (default: %(default)s)')

    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/",
                        help='Directory path to the dataset. (default: %(default)s)')

    parser.add_argument('--disable_evaluation', dest='enable_evaluation',
                        action='store_false',
                        help='Boolean flag indicating if evaluation is disabled.')

    parser.add_argument('--disable_validation', dest='enable_validation',
                        action='store_false',
                        help='Boolean flag indicating if validation is disabled.')

    parser.add_argument('--epoch', dest='epoch', default=1,
                        type=check_int_positive,
                        help='The number of epochs used in training models. (default: %(default)s)')

    parser.add_argument('--lambda_l2', dest='lamb_l2', default=1.0,
                        type=check_float_positive,
                        help='L2 Regularizer strength used in models. (default: %(default)s)')

    parser.add_argument('--lambda_keyphrase', dest='lamb_keyphrase', default=1.0,
                        type=check_float_positive,
                        help='Keyphrase loss strength used in models. (default: %(default)s)')

    parser.add_argument('--lambda_latent', dest='lamb_latent', default=5.0,
                        type=check_float_positive,
                        help='Latent reconstruction loss strength used in models. (default: %(default)s)')

    parser.add_argument('--lambda_rating', dest='lamb_rating', default=1.0,
                        type=check_float_positive,
                        help='Rating loss strength used in models. (default: %(default)s)')

    parser.add_argument('--learning_rate', dest='learning_rate', default=0.0001,
                        type=check_float_positive,
                        help='Learning rate used in training models. (default: %(default)s)')

    parser.add_argument('--model', dest='model', default="CE-VAE",
                        help='Model currently using. (default: %(default)s)')

    parser.add_argument('--optimizer', dest='optimizer', default="Adam",
                        help='Optimizer currently using. (default: %(default)s)')

    parser.add_argument('--predict_batch_size', dest='predict_batch_size', default=128,
                        type=check_int_positive,
                        help='Batch size used in prediction. (default: %(default)s)')

    parser.add_argument('--rank', dest='rank', default=200,
                        type=check_int_positive,
                        help='Latent dimension. (default: %(default)s)')

    parser.add_argument('--test', dest='test_set', default="Rtest.npz",
                        help='Test set sparse matrix. (default: %(default)s)')

    parser.add_argument('--test_keyphrase', dest='test_keyphrase_set', default="Rtest_keyphrase.npz",
                        help='Test keyphrase sparse matrix. (default: %(default)s)')

    parser.add_argument('--topk', dest='topk', default=10,
                        type=check_int_positive,
                        help='The number of items being recommended at top. (default: %(default)s)')

    parser.add_argument('--train', dest='train_set', default="Rtrain.npz",
                        help='Train set sparse matrix. (default: %(default)s)')

    parser.add_argument('--train_batch_size', dest='train_batch_size', default=128,
                        type=check_int_positive,
                        help='Batch size used in training. (default: %(default)s)')

    parser.add_argument('--train_keyphrase', dest='train_keyphrase_set', default="Rtrain_keyphrase.npz",
                        help='Train keyphrase sparse matrix. (default: %(default)s)')

    parser.add_argument('--valid', dest='valid_set', default="Rvalid.npz",
                        help='Valid set sparse matrix. (default: %(default)s)')

    parser.add_argument('--valid_keyphrase', dest='valid_keyphrase_set', default="Rvalid_keyphrase.npz",
                        help='Valid keyphrase sparse matrix. (default: %(default)s)')

    args = parser.parse_args()

    main(args)

