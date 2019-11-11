from evaluation.general_performance import average_precisionk
from tqdm import tqdm
from utils.critique import critique_keyphrase

import numpy as np
import pandas as pd


def critiquing_evaluation(train_set, keyphrase_train_set, item_keyphrase_train_set, model, model_name, num_users, num_items, num_users_sampled, critiquing_function, topk, valid_set):
    fmap_results = [[] for _ in topk]
    #sampled_users = np.random.choice(num_users, num_users_sampled)
    sampled_users = np.random.choice(num_users, num_users)
    for user in tqdm(sampled_users):
        user_valid = valid_set[user].toarray().nonzero()[1]
        top_items_before_critique, top_items_after_critique, affected_items = critique_keyphrase(train_set,
                                                                                                 keyphrase_train_set,
                                                                                                 item_keyphrase_train_set,
                                                                                                 model,
                                                                                                 user,
                                                                                                 num_items,
                                                                                                 critiquing_function,
                                                                                                topk_keyphrase=10)
        # This code is for original MAP
        # answers = user_valid

        # This code is for FMAP
        training_input = train_set[user].nonzero()[1]
        all_items = np.array(range(num_items))
        #remove items used in training from answers
        possible_items = all_items[~np.in1d(all_items, training_input)]
        answers = np.intersect1d(affected_items, possible_items)

        # This code is for updated Falling MAP(new version)
        # answers = np.intersect1d(user_valid, affected_items)
        
        #This code is for increasing MAP(new version)
        # all_items = np.array(range(num_items))
        # unaffected_items = all_items[~np.in1d(all_items, affected_items)]
        # answers = np.intersect1d(unaffected_items, user_valid)
        
        for i, k in enumerate(topk):
            fmap_results[i].append( average_precisionk(top_items_before_critique[:k],
                                                      np.isin(top_items_before_critique[:k], answers))
                                   - average_precisionk(top_items_after_critique[:k],
                                                        np.isin(top_items_after_critique[:k], answers)))

    fmap_results_dict = dict()
    for i, k in enumerate(topk):
        fmap_results_dict['F-MAP@{0}'.format(k)] = fmap_results[i]
    df_fmap = pd.DataFrame(fmap_results_dict)

    return df_fmap

