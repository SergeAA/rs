import numpy as np


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    flags = np.isin(bought_list, recommended_list)
    hit_rate = (flags.sum() > 0) * 1
    return hit_rate


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)
    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:k])
    flags = np.isin(recommended_list, bought_list) * 1
    return (flags * prices_recommended).sum() / prices_recommended.sum()


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    prices_bought = np.array(prices_bought)
    recommended_list = np.array(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:k])

    flags = np.isin(recommended_list, bought_list) * 1
    revenue_recommended = (flags * prices_recommended).sum()

    return revenue_recommended / prices_bought.sum()