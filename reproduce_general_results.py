from experiment.general import general
from utils.io import load_numpy

import argparse


def main(args):

    params = dict()
    params['tuning_result_path'] = args.tuning_result_path

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    R_test = load_numpy(path=args.data_dir, name=args.test_set)
    R_train_keyphrase = load_numpy(path=args.data_dir, name=args.train_keyphrase_set)
    R_test_keyphrase = load_numpy(path=args.data_dir, name=args.test_keyphrase_set)
    R_train_keyphrase[R_train_keyphrase != 0] = 1
    R_test_keyphrase[R_test_keyphrase != 0] = 1

    general(R_train,
            R_test,
            R_train_keyphrase.todense(),
            R_test_keyphrase,
            params,
            save_path=args.save_path,
            final_explanation=args.final_explanation)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce Final General Recommendation Performance")

    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/")
    parser.add_argument('--final_explanation', dest='final_explanation', action="store_true")
    parser.add_argument('--save_path', dest='save_path', default="beer_final/cdevae_final.csv")
    parser.add_argument('--test', dest='test_set', default="Rtest.npz")
    parser.add_argument('--test_keyphrase', dest='test_keyphrase_set', default="Rtest_keyphrase.npz",
                        help='Test keyphrase sparse matrix. (default: %(default)s)')
    parser.add_argument('--train', dest='train_set', default="Rtrain.npz")
    parser.add_argument('--train_keyphrase', dest='train_keyphrase_set', default="Rtrain_keyphrase.npz",
                        help='Train keyphrase sparse matrix. (default: %(default)s)')
    parser.add_argument('--tuning_result_path', dest='tuning_result_path', default="beer")

    args = parser.parse_args()

    main(args)

