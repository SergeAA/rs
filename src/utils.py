import numpy as np


def prefilter_items(data, price=(1, 100), popular=(0.01, 0.5), week_min=12*4, product_filter=None, top_n=None):
    stat = data.groupby('item_id').agg({
        'user_id': 'nunique',
        'sales_value': 'median',
        'quantity': 'median',
        'week_no': 'max'
    }).reset_index()

    stat['share_unique_users'] = stat['user_id'] / data['user_id'].nunique()
    stat['item_price'] = stat['sales_value'] / stat['quantity']

    flt = stat['user_id'] == 0

    if price[0] is not None:
        flt |= stat['sales_value'] < price[0]  # дешевые товары - менее чем 1 $
        flt |= stat['item_price'] < price[0]  # дешевая цена за единицу товара
    if price[1] is not None:
        flt |= stat['sales_value'] > price[1]  # дорогие товары - более чем 100$

    if popular[0] is not None:
        flt |= stat['share_unique_users'] < popular[0]  # самые непопулярные товары
    if popular[1] is not None:
        flt |= stat['share_unique_users'] > popular[1]  # самые популярные товары

    if week_min is not None:
        flt |= stat['week_no'] < week_min  # не продавались за последние 12 месяцев

    stat = stat[~flt]

    if product_filter is not None:
        stat = stat.merge(product_filter, how='inner', on='item_id')

    stat['weight'] = stat['quantity'] * stat['item_price'] * stat['share_unique_users'] * stat['week_no']
    stat['weight'] /= stat['weight'].max()

    if top_n is not None:
        stat = stat.sort_values('weight', ascending=False).head(top_n)

    data = data.merge(stat[['item_id', 'weight']], how='left', on='item_id')
    data.loc[data['weight'].isnull(), 'item_id'] = 999999
    data.loc[data['weight'].isnull(), 'weight'] = 0.0

    return data


def postfilter_items(user_id, recommednations):
    pass
