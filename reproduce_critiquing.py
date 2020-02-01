from experiment.critiquing import critiquing
from utils.argcheck import check_int_positive
from utils.io import load_numpy

import argparse
import numpy as np
import scipy.sparse as sparse


def main(args):

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    R_train_keyphrase = load_numpy(path=args.data_dir, name=args.train_keyphrase_set).toarray()

    R_train_keyphrase[R_train_keyphrase != 0] = 1
    R_train_item_keyphrase = load_numpy(path=args.data_dir, name=args.train_item_keyphrase_set).T.toarray()

    num_items, num_keyphrases = R_train_item_keyphrase.shape
    for item in range(num_items):
        item_keyphrase = R_train_item_keyphrase[item]
        nonzero_keyphrases_index = item_keyphrase.nonzero()[0]
        nonzero_keyphrases_frequency = item_keyphrase[nonzero_keyphrases_index]
        candidate_index = nonzero_keyphrases_index[np.argsort(-nonzero_keyphrases_frequency)[:10]]
        binarized_keyphrase = np.zeros(num_keyphrases)
        binarized_keyphrase[candidate_index] = 1
        R_train_item_keyphrase[item] = binarized_keyphrase

    R_train_item_keyphrase = sparse.csr_matrix(R_train_item_keyphrase).T

    params = dict()
#    params['model_saved_path'] = args.model_saved_path

    critiquing(R_train,
               R_train_keyphrase,
               R_train_item_keyphrase,
               params,
               args.num_users_sampled,
               load_path=args.load_path,
               save_path=args.save_path,
               critiquing_function=args.critiquing_function)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce Critiquing Performance")

    parser.add_argument('--critiquing_function', dest='critiquing_function', default="energy_redistribution")
    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/")
    parser.add_argument('--load_path', dest='load_path', default="critiquing_hyperparameters/beer/hyper_parameters.csv")
#    parser.add_argument('--model_saved_path', dest='model_saved_path', default="beer")
    parser.add_argument('--num_users_sampled', dest='num_users_sampled', type=check_int_positive, default=500)
    parser.add_argument('--save_path', dest='save_path', default="beer_fmap/beer_Critiquing")
    parser.add_argument('--train', dest='train_set', default="Rtrain.npz")
    parser.add_argument('--train_item_keyphrase', dest='train_item_keyphrase_set', default="Rtrain_item_keyphrase.npz",
                        help='Train keyphrase sparse matrix. (default: %(default)s)')
    parser.add_argument('--train_keyphrase', dest='train_keyphrase_set', default="Rtrain_keyphrase.npz",
                        help='Train keyphrase sparse matrix. (default: %(default)s)')

    args = parser.parse_args()

    main(args)

