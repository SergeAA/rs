import pandas as pd


def load_data():

    TRAINTEST = '../data/retail_train.csv'
    PRODUCT = '../data/product.csv'
    BASELINE = '../data/predictions_basic.pkl'

    data = pd.read_csv(TRAINTEST)
    test_size_weeks = 3
    train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
    test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

    product = pd.read_csv(PRODUCT)
    product.columns = [col.lower() for col in product.columns]
    product.rename(columns={'product_id': 'item_id'}, inplace=True)

    baseline = pd.read_pickle(BASELINE)

    return train, test, product, baseline
