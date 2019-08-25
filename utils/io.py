from scipy.sparse import save_npz, load_npz


def load_numpy(path, name):
    return load_npz(path+name).tocsr()
