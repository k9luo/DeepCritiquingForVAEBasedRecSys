from experiment.tuning import hyper_parameter_tuning
from utils.io import load_numpy, load_yaml
from utils.modelnames import models

import argparse


def main(args):
    params = load_yaml(args.parameters)

    params['models'] = {params['models']: models[params['models']]}

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    R_valid = load_numpy(path=args.data_dir, name=args.valid_set)

    hyper_parameter_tuning(R_train, R_valid, params, save_path=args.save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/")
    parser.add_argument('--parameters', dest='parameters', default='config/default.yml')
    parser.add_argument('--save_path', dest='save_path', default="cdevae_tuning.csv")
    parser.add_argument('--train', dest='train_set', default="Rtrain.npz")
    parser.add_argument('--valid', dest='valid_set', default="Rvalid.npz")

    args = parser.parse_args()

    main(args)
