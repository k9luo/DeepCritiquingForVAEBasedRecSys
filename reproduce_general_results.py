from experiment.general import general
from utils.io import load_numpy

import argparse


def main(args):

    params = dict()
    params['tuning_result_path'] = args.tuning_result_path

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    R_test = load_numpy(path=args.data_dir, name=args.test_set)

    general(R_train, R_test, params, save_path=args.save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce Final General Recommendation Performance")

    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/")
    parser.add_argument('--save_path', dest='save_path', default="beer_final/cdevae_final.csv")
    parser.add_argument('--test', dest='test_set', default="Rtest.npz")
    parser.add_argument('--train', dest='train_set', default="Rtrain.npz")
    parser.add_argument('--tuning_result_path', dest='tuning_result_path', default="beer")

    args = parser.parse_args()

    main(args)

