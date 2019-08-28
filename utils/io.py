from scipy.sparse import save_npz, load_npz

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
