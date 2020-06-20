import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from .utils import FAKE_ITEM, unique


class RecommenderDataset:
    """Базовый класс для удобства работы с данными

    Input
    -----
    data: pd.DataFrame
        DataFrame описывающий транзакции между user / item
    values: str
        поле по котором будет происходит агрегация и сортировка (сила взаимодействия user/item)
        по умолчанию `weight`
    aggfunc: str
        способ агрегации поля values (по умолчанию `mean`)
    weighting: str
        способ взвешивания матрицы user_item (`bm25` или `tfidf`)
        по умолчанию `bm25`
    """

    def __init__(self, data, values='weight', aggfunc='count', weighting='bm25'):

        self.users_top = data[data['item_id'] != FAKE_ITEM].groupby(
            ['user_id', 'item_id']).agg({values: aggfunc}).reset_index()
        self.users_top.sort_values(values, ascending=False, inplace=True)

        self.top = data[data['item_id'] != FAKE_ITEM].groupby('item_id').agg({values: aggfunc}).reset_index()
        self.top = self.top.sort_values(values, ascending=False).item_id.tolist()

        user_item_matrix = pd.pivot_table(data, index='user_id', columns='item_id',
                                          values=values, aggfunc=aggfunc,
                                          fill_value=0).astype(float)

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values
        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))
        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))

        self.FAKE_ITEM_ID = self.itemid_to_id[FAKE_ITEM]
        self.__userTop = {}

        if weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(user_item_matrix.T).T
        elif weighting == 'bm25':
            self.user_item_matrix = bm25_weight(user_item_matrix.T).T
        else:
            self.user_item_matrix = user_item_matrix

        self.csr_matrix = csr_matrix(self.user_item_matrix).T.tocsr()

    def userExist(self, userId):
        return self.userid_to_id.get(userId, None) is not None

    def userTop(self, userId, N=50):
        if not self.userExist(userId):
            return []

        if self.__userTop.get(userId, None) is None:
            self.__userTop[userId] = self.users_top[self.users_top.user_id == userId].head(200).item_id.tolist()

        return self.__userTop[userId][:N]

    def extend(self, res, N=5):
        """дополняет результат до нужного количества из TOP товаров"""
        res = unique(list(res))

        if len(res) < N:
            res.extend(self.top[:N])
            res = unique(res)
            res = res[:N]

        assert len(res) == N, f'Количество рекомендаций != {N}'
        return res


class BaseRecommender:
    """Базовый класс для наследования
       Модель, которая всегда рекомендует полулярные товары

    Input
    -----
    ds: RecommenderDataset
        подготовленный RecommenderDataset обьект
    """

    def __init__(self, ds):
        assert isinstance(ds, RecommenderDataset), 'Нужен обьект типа RecommenderDataset'
        self.ds = ds

    def fit(self):
        return self

    def extend(self, res, N=5, userId=None):
        res = list(res)
        if (len(res) < N) and (userId is not None):
            res.extend(self.ds.userTop(userId, N))
            res = unique(res)
            res = res[:N]
        return self.ds.extend(res, N)

    def _recommend(self, userId=None, N=5):
        return self.extend([], N=N, userId=userId)

    def _similarItems(self, userId=None, N=5):
        return self._recommend(userId, N)

    def _similarUsers(self, userId=None, N=5):
        return self._recommend(userId, N)

    def recommend(self, users, N=5, by='model'):
        if by == 'similarItems':
            func = self._similarItems
        elif by == 'similarUsers':
            func = self._similarUsers
        else:
            func = self._recommend
        return [func(user, N) for user in users]


class OwnRecommender(BaseRecommender):
    """Модель, которая рекомендует товары, среди товаров, купленных юзером

    Input
    -----
    ds: RecommenderDataset
        подготовленный RecommenderDataset обьект
    """

    def fit(self):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
        self.model = ItemItemRecommender(K=1, num_threads=4)
        self.model.fit(self.ds.csr_matrix)

        return self

    def _recommend(self, userId, N=5):
        """Рекомендуем товары для пользователя обученной моделью"""
        if not self.ds.userExist(userId):
            return self.ds.extend([], N)

        params = {
            'userid': self.ds.userid_to_id[userId],
            'user_items': self.ds.csr_matrix,
            'N': N,
            'filter_already_liked_items': False,
            'filter_items': [self.ds.FAKE_ITEM_ID],
            'recalculate_user': True
        }

        res = [
            self.ds.id_to_itemid[rec[0]]
            for rec in self.model.recommend(**params)
        ]

        return self.extend(res, N)


class AlsRecommender(OwnRecommender):
    """Модель, обученная ALS

    Input
    -----
    ds: RecommenderDataset
        подготовленный RecommenderDataset обьект
    """

    def fit(self, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        self.model = AlternatingLeastSquares(factors=n_factors,
                                             regularization=regularization,
                                             iterations=iterations,
                                             num_threads=num_threads)
        self.model.fit(self.ds.csr_matrix)

        return self

    def _similarItems(self, userId, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        if not self.ds.userExist(userId):
            return self.ds.extend([], N)

        def _get_similar_item(item_id):
            """Находит товар, похожий на item_id"""
            recs = self.model.similar_items(self.ds.itemid_to_id[item_id], N=2)
            if len(recs) > 1:
                top_rec = recs[1][0]
                return self.ds.id_to_itemid[top_rec]
            return item_id

        res = [_get_similar_item(item) for item in self.ds.userTop(userId, N)]
        return self.extend(res, N)

    def _similarUsers(self, userId, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        if not self.ds.userExist(userId):
            return self.ds.extend([], N)

        res = []
        similar_users = [rec[0] for rec in self.model.similar_users(self.ds.userid_to_id[userId], N=N+1)]
        similar_users = similar_users[1:]

        for user in similar_users:
            res.extend(self.ds.userTop(userId, 1))

        return self.extend(res, N)
