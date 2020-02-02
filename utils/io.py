from scipy.sparse import load_npz

import ast
import os
import pandas as pd
import yaml


def load_numpy(path, name):
    return load_npz(path+name).tocsr()

def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)

def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)

def save_dataframe_csv(df, path, name):
    df.to_csv(path+name, index=False)

def find_best_hyperparameters(folder_path, metric):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df = df.loc[df['lambda_latent'] == df['lambda_keyphrase']]
        df[metric+'_Score'] = df[metric].map(lambda x: ast.literal_eval(x)[0])
        best_settings.append(df.loc[df[metric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings)

    if any(df['model'].duplicated()):
        df = df.groupby('model', group_keys=False).apply(lambda x: x.loc[x[metric+'_Score'].idxmax()]).reset_index(drop=True)

    df = df.drop(metric+'_Score', axis=1)

    return df
