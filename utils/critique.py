import numpy as np
import pandas as pd


def critique_keyphrase(train_set, keyphrase_train_set, item_keyphrase_train_set, model, user_index, num_items, energy_df, topk_keyphrase=10):
    # Get rating and explanation prediction for the given user and all item pairs
    rating, explanation = model.predict(train_set[user_index].todense())
#    print(sum(explanation[0]))
    energy_df = energy_df.append(pd.DataFrame([{"User": user_index, "Time": 1, "Energy": sum(explanation[0])}]))

    # For the given user, get top k keyphrases for that user
    explanation_rank_list = np.argsort(-explanation[0])[:topk_keyphrase]

    keyphrase_index = []
    # Random critique one keyphrase among existing predicted keyphrases
    critiqued_keyphrase_index = int(np.random.choice(explanation_rank_list, 1)[0])
    keyphrase_index.append(critiqued_keyphrase_index)

    # Get all affected items
#    affected_items = item_keyphrase_train_set[keyphrase_index].nonzero()[1]

    """
    # Redistribute keyphrase prediction score
    rating_difference = explanation[0][keyphrase_index]
    explanation[0][keyphrase_index] = 0
    keyphrase_ratio = explanation / sum(explanation[0])
    keyphrase_redistribute_score = keyphrase_ratio * rating_difference
    explanation += keyphrase_redistribute_score
    """

    explanation[0][keyphrase_index] = min(explanation[0])
#    explanation[0][keyphrase_index] = -1


    for t in range(4):
        modified_rating, modified_explanation = model.refine_predict(train_set[user_index].todense(), explanation)
#        print(sum(modified_explanation[0]))
        energy_df = energy_df.append(pd.DataFrame([{"User": user_index, "Time": t+2, "Energy": sum(modified_explanation[0])}]))


        explanation_rank_list = np.argsort(-modified_explanation[0])[:topk_keyphrase]
#        import ipdb; ipdb.set_trace()
        explanation_rank_list = np.setdiff1d(explanation_rank_list, keyphrase_index)
        critiqued_keyphrase_index = int(np.random.choice(explanation_rank_list, 1)[0])
        keyphrase_index = np.append(keyphrase_index, critiqued_keyphrase_index)
#        print(keyphrase_index)
        keyphrase_index = np.unique(keyphrase_index)
        affected_items = item_keyphrase_train_set[keyphrase_index].nonzero()[1]
        modified_explanation[0][keyphrase_index] = min(modified_explanation[0])
#        modified_explanation[0][keyphrase_index] = -1

        explanation = modified_explanation

    return np.argsort(rating.flatten())[::-1], np.argsort(modified_rating.flatten())[::-1], affected_items, energy_df

