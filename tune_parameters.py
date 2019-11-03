from experiment.tuning import hyper_parameter_tuning
from utils.io import load_numpy, load_yaml
from utils.modelnames import models

import argparse


def main(args):
    params = load_yaml(args.parameters)
    
    params['models'] = {params['models']: models[params['models']]}

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    R_valid = load_numpy(path=args.data_dir, name=args.valid_set)
    R_train_keyphrase = load_numpy(path=args.data_dir, name=args.train_keyphrase_set)
    R_valid_keyphrase = load_numpy(path=args.data_dir, name=args.valid_keyphrase_set)
    R_train_keyphrase[R_train_keyphrase != 0] = 1
    R_valid_keyphrase[R_valid_keyphrase != 0] = 1

    hyper_parameter_tuning(R_train,
                           R_valid,
                           R_train_keyphrase.todense(),
                           R_valid_keyphrase,
                           params,
                           save_path=args.save_path,
                           tune_explanation=args.tune_explanation)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/")
    parser.add_argument('--parameters', dest='parameters', default='config/default.yml')
    parser.add_argument('--save_path', dest='save_path', default="cdevae_tuning.csv")
    parser.add_argument('--train', dest='train_set', default="Rtrain.npz")
    parser.add_argument('--train_keyphrase', dest='train_keyphrase_set', default="Rtrain_keyphrase.npz",
                        help='Train keyphrase sparse matrix. (default: %(default)s)')
    parser.add_argument('--tune_explanation', dest='tune_explanation', action="store_true")
    parser.add_argument('--valid', dest='valid_set', default="Rvalid.npz")
    parser.add_argument('--valid_keyphrase', dest='valid_keyphrase_set', default="Rvalid_keyphrase.npz",
                        help='Valid keyphrase sparse matrix. (default: %(default)s)')

    args = parser.parse_args()

    main(args)

