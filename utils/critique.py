import numpy as np


def critique_keyphrase(train_set, keyphrase_train_set, item_keyphrase_train_set, model, user_index, num_items, critiquing_function, topk_keyphrase=10):
    # Get rating and explanation prediction for the given user and all item pairs
    rating, explanation = model.predict(train_set[user_index].todense())

    # For the given user, get top k keyphrases for that user
    explanation_rank_list = np.argsort(-explanation[0])[:topk_keyphrase]

    # Random critique one keyphrase among existing predicted keyphrases
    keyphrase_index = int(np.random.choice(explanation_rank_list, 1)[0])

    # Get all affected items
    affected_items = item_keyphrase_train_set[keyphrase_index].nonzero()[1]

    if critiquing_function == 'upper_bound':
        after_critique = rating.flatten()
        after_critique[affected_items] = -999999
        return np.argsort(rating.flatten())[::-1], np.argsort(after_critique)[::-1], affected_items
        #return np.argsort(rating.flatten())[::-1], np.append(np.delete(np.argsort(rating.flatten())[::-1], affected_items), affected_items), affected_items

    if critiquing_function == 'energy_redistribution':
        # Redistribute keyphrase prediction score
        rating_difference = explanation[0][keyphrase_index]
        explanation[0][keyphrase_index] = 0
        keyphrase_ratio = explanation / sum(explanation[0])
        keyphrase_redistribute_score = keyphrase_ratio * rating_difference
        explanation += keyphrase_redistribute_score
    elif critiquing_function == 'zero_out':
        explanation[0][keyphrase_index] = min(explanation[0])
    else:
        raise ValueError("The given critiquing function is not implemented!")

    modified_rating, modified_explanation = model.refine_predict(train_set[user_index].todense(), explanation)

    return np.argsort(rating.flatten())[::-1], np.argsort(modified_rating.flatten())[::-1], affected_items

