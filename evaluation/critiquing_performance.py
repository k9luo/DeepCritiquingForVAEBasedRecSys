from evaluation.general_performance import average_precisionk
from tqdm import tqdm
from utils.critique import critique_keyphrase

import numpy as np
import pandas as pd


def critiquing_evaluation(train_set, keyphrase_train_set, item_keyphrase_train_set, model, model_name, num_users, num_items, num_users_sampled, critiquing_function, topk):
    fmap_results = [[] for _ in topk]
    sampled_users = np.random.choice(num_users, num_users_sampled)
    for user in tqdm(sampled_users):
        top_items_before_critique, top_items_after_critique, affected_items = critique_keyphrase(train_set,
                                                                                                 keyphrase_train_set,
                                                                                                 item_keyphrase_train_set,
                                                                                                 model,
                                                                                                 user,
                                                                                                 num_items,
                                                                                                 critiquing_function,
                                                                                                 topk_keyphrase=10)

        all_items = np.array(range(num_items))
        unaffected_items = all_items[~np.in1d(all_items, affected_items)]

        for i, k in enumerate(topk):
            fmap_results[i].append(average_precisionk(top_items_before_critique[:k],
                                                      np.isin(top_items_before_critique[:k], affected_items))
                                   - average_precisionk(top_items_after_critique[:k],
                                                        np.isin(top_items_after_critique[:k], affected_items)))

    fmap_results_dict = dict()
    for i, k in enumerate(topk):
        fmap_results_dict['F-MAP@{0}'.format(k)] = fmap_results[i]
    df_fmap = pd.DataFrame(fmap_results_dict)

    return df_fmap

