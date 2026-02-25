# 使用ALS进行矩阵分解
# 说明：本文件手写矩阵运算与ALS推荐算法，方便学习流程，非高性能实现。
from itertools import product, chain
from copy import deepcopy


class Matrix(object):
    # 简单矩阵类：用列表保存数据，提供转置、单位阵、逆矩阵、矩阵乘等基础操作。
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def row(self, row_no):
        """获取矩阵的某一行。
        参数:
            row_no {int} -- 行号（从0开始）
        返回:
            Matrix
        """

        return Matrix([self.data[row_no]])

    def col(self, col_no):
        """获取矩阵的某一列。
        参数:
            col_no {int} -- 列号（从0开始）
        返回:
            Matrix
        """
        m = self.shape[0]
        return Matrix([[self.data[i][col_no]] for i in range(m)])

    @property
    def is_square(self):
        """判断是否为方阵。
        返回:
            bool
        """

        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        """返回转置矩阵。
        返回:
            Matrix
        """

        data = list(map(list, zip(*self.data)))
        return Matrix(data)

    def _eye(self, n):
        """生成 n×n 单位矩阵（列表形式）。"""

        return [[0 if i != j else 1 for j in range(n)] for i in range(n)]

    @property
    def eye(self):
        """生成与自身同阶的单位矩阵。
        返回:
            Matrix
        """

        assert self.is_square, "The matrix has to be square!"
        data = self._eye(self.shape[0])
        return Matrix(data)

    def _gaussian_elimination(self, aug_matrix):
        """高斯消元：把增广矩阵左侧化为单位阵，右侧得到逆矩阵"""

        n = len(aug_matrix)
        m = len(aug_matrix[0])

        # 自上而下做前向消元，得到上三角
        for col_idx in range(n):
            # 若对角线元素为0，则向下寻找非零行相加，避免除零
            if aug_matrix[col_idx][col_idx] == 0:
                row_idx = col_idx
                while row_idx < n and aug_matrix[row_idx][col_idx] == 0:
                    row_idx += 1
                for i in range(col_idx, m):
                    aug_matrix[col_idx][i] += aug_matrix[row_idx][i]

            # 消去该列其他行的非零元素
            for i in range(col_idx + 1, n):
                if aug_matrix[i][col_idx] == 0:
                    continue
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in range(col_idx, m):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # 自下而上回代，得到对角阵
        for col_idx in range(n - 1, -1, -1):
            # 消去上方非零元素
            for i in range(col_idx):
                if aug_matrix[i][col_idx] == 0:
                    continue
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in chain(range(i, col_idx + 1), range(n, m)):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # 把对角线归一化
        for i in range(n):
            k = 1 / aug_matrix[i][i]
            aug_matrix[i][i] *= k
            for j in range(n, m):
                aug_matrix[i][j] *= k

        return aug_matrix

    def _inverse(self, data):
        """计算逆矩阵：把原矩阵与单位阵拼成增广矩阵，做高斯消元"""

        n = len(data)
        unit_matrix = self._eye(n)
        aug_matrix = [a + b for a, b in zip(self.data, unit_matrix)]
        ret = self._gaussian_elimination(aug_matrix)
        return list(map(lambda x: x[n:], ret))

    @property
    def inverse(self):
        """获取自身的逆矩阵。
        返回:
            Matrix
        """

        assert self.is_square, "The matrix has to be square!"
        data = self._inverse(self.data)
        return Matrix(data)

    def _row_mul(self, row_A, row_B):
        """两个等长向量对应元素相乘再求和（点积）。"""

        return sum(x[0] * x[1] for x in zip(row_A, row_B))

    def _mat_mul(self, row_A, B):
        """矩阵乘法的行向量辅助计算，返回一行结果。"""

        row_pairs = product([row_A], B.transpose.data)
        return [self._row_mul(*row_pair) for row_pair in row_pairs]

    def mat_mul(self, B):
        """矩阵乘法。
        参数:
            B {Matrix}
        返回:
            Matrix
        """

        error_msg = "A's column count does not match B's row count!"
        assert self.shape[1] == B.shape[0], error_msg
        return Matrix([self._mat_mul(row_A, B) for row_A in self.data])

    def _mean(self, data):
        """计算二维数组每列的均值，返回列表。"""

        m = len(data)
        n = len(data[0])
        ret = [0 for _ in range(n)]
        for row in data:
            for j in range(n):
                ret[j] += row[j] / m
        return ret

    def mean(self):
        """计算均值并包装为 Matrix。"""

        return Matrix(self._mean(self.data))

    def scala_mul(self, scala):
        """标量乘法：全矩阵元素乘以常数。
        参数:
            scala {float}
        返回:
            Matrix
        """

        m, n = self.shape
        data = deepcopy(self.data)
        for i in range(m):
            for j in range(n):
                data[i][j] *= scala
        return Matrix(data)


import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os


class ALS(object):
    # 交替最小二乘（ALS）实现：把评分矩阵分解为用户矩阵U和物品矩阵V，交替求解。
    def __init__(self):
        self.user_ids = None
        self.item_ids = None
        self.user_ids_dict = None
        self.item_ids_dict = None
        self.user_matrix = None
        self.item_matrix = None
        self.user_items = None
        self.shape = None
        self.rmse = None

    def _process_data(self, X):
        """将评分矩阵X转化为稀疏矩阵
        输入参数X:
            X {list} -- 2d list with int or float(user_id, item_id, rating)
        输出结果:
            dict -- {user_id: {item_id: rating}}
            dict -- {item_id: {user_id: rating}}
        """
        # 收集唯一用户与物品ID
        self.user_ids = tuple((set(map(lambda x: x[0], X))))
        self.user_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.user_ids)))

        self.item_ids = tuple((set(map(lambda x: x[1], X))))
        self.item_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.item_ids)))

        self.shape = (len(self.user_ids), len(self.item_ids))

        ratings = defaultdict(lambda: defaultdict(int))
        ratings_T = defaultdict(lambda: defaultdict(int))
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating

        err_msg = "Length of user_ids %d and ratings %d not match!" % (
            len(self.user_ids),
            len(ratings),
        )
        assert len(self.user_ids) == len(ratings), err_msg

        err_msg = "Length of item_ids %d and ratings_T %d not match!" % (
            len(self.item_ids),
            len(ratings_T),
        )
        assert len(self.item_ids) == len(ratings_T), err_msg
        return ratings, ratings_T

    def _users_mul_ratings(self, users, ratings_T):
        """用户矩阵（稠密）与评分矩阵（稀疏）相乘。
        结果是 k×n 物品矩阵（n 为物品数）。
        参数:
            users {Matrix} -- k×m 用户矩阵，m 为用户数
            ratings_T {dict} -- 物品被哪些用户评分的稀疏表示 {item_id: {user_id: rating}}
        返回:
            Matrix -- 物品矩阵
        """

        def f(users_row, item_id):
            # 只遍历该物品被评分过的用户，避免全矩阵计算
            user_ids = iter(ratings_T[item_id].keys())
            scores = iter(ratings_T[item_id].values())
            col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
            _users_row = map(lambda x: users_row[x], col_nos)
            return sum(a * b for a, b in zip(_users_row, scores))

        ret = [
            [f(users_row, item_id) for item_id in self.item_ids]
            for users_row in users.data
        ]
        return Matrix(ret)

    def _items_mul_ratings(self, items, ratings):
        """物品矩阵（稠密）与评分矩阵（稀疏）相乘。
        结果是 k×m 用户矩阵（m 为用户数）。
        参数:
            items {Matrix} -- k×n 物品矩阵，n 为物品数
            ratings {dict} -- 用户对物品的评分稀疏表示 {user_id: {item_id: rating}}
        返回:
            Matrix -- 用户矩阵
        """

        def f(items_row, user_id):
            # 同样仅遍历该用户评分过的物品
            item_ids = iter(ratings[user_id].keys())
            scores = iter(ratings[user_id].values())
            col_nos = map(lambda x: self.item_ids_dict[x], item_ids)
            _items_row = map(lambda x: items_row[x], col_nos)
            return sum(a * b for a, b in zip(_items_row, scores))

        ret = [
            [f(items_row, user_id) for user_id in self.user_ids]
            for items_row in items.data
        ]
        return Matrix(ret)

    # 生成随机矩阵
    def _gen_random_matrix(self, n_rows, n_colums):
        # print(n_colums, ' ', n_rows)
        # data = [[random() for _ in range(n_colums)] for _ in range(n_rows)]
        # d = 2
        data = np.random.rand(n_rows, n_colums)
        return Matrix(data)

    # 计算RMSE
    def _get_rmse(self, ratings):
        m, n = self.shape
        mse = 0.0
        n_elements = sum(map(len, ratings.values()))
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                item_id = self.item_ids[j]
                rating = ratings[user_id][item_id]
                if rating > 0:
                    # 只对有评分的位置计算预测误差
                    user_row = self.user_matrix.col(i).transpose
                    item_col = self.item_matrix.col(j)
                    rating_hat = user_row.mat_mul(item_col).data[0][0]
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error / n_elements
        return mse**0.5

    # 模型训练
    def fit(self, X, k, max_iter=10):
        ratings, ratings_T = self._process_data(X)
        self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
        m, n = self.shape

        error_msg = "参数 k 需小于原矩阵的秩（行列最小值）"
        assert k < min(m, n), error_msg

        self.user_matrix = self._gen_random_matrix(k, m)

        for i in range(max_iter):
            # 奇数轮：固定物品矩阵，求用户矩阵
            if i % 2:
                items = self.item_matrix
                self.user_matrix = self._items_mul_ratings(
                    items.mat_mul(items.transpose).inverse.mat_mul(items), ratings
                )
            # 偶数轮：固定用户矩阵，求物品矩阵
            else:
                users = self.user_matrix
                self.item_matrix = self._users_mul_ratings(
                    users.mat_mul(users.transpose).inverse.mat_mul(users), ratings_T
                )
            rmse = self._get_rmse(ratings)
            print("迭代: %d, RMSE: %.6f" % (i + 1, rmse))

        self.rmse = rmse

    # Top-n推荐，用户列表：user_id, n_items: Top-n
    def _predict(self, user_id, n_items):
        users_col = self.user_matrix.col(self.user_ids_dict[user_id])
        users_col = users_col.transpose

        items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
        items_scores = map(lambda x: (self.item_ids[x[0]], x[1]), items_col)
        viewed_items = self.user_items[user_id]
        # 过滤掉用户已看过的物品，只推荐未看过的
        items_scores = filter(lambda x: x[0] not in viewed_items, items_scores)

        return sorted(items_scores, key=lambda x: x[1], reverse=True)[:n_items]

    # 预测多个用户
    def predict(self, user_ids, n_items=10):
        return [self._predict(user_id, n_items) for user_id in user_ids]


# 以下是自己写的部分
def format_prediction(item_id, score):
    return "物品ID:%d 预测评分:%.2f" % (item_id, score)


def load_movie_ratings(file_name):
    # 读取评分文件，格式：userId,movieId,rating,timestamp
    # 这里只取前三列，跳过时间戳
    base_dir = os.path.dirname(__file__)  # 以当前脚本所在目录为基准
    file_path = os.path.join(base_dir, file_name)
    f = open(file_path)
    lines = iter(f)
    col_names = ", ".join(next(lines)[:-1].split(",")[:-1])
    print("列名为: %s。" % col_names)
    data = [
        [
            float(x) if i == 2 else int(x)
            for i, x in enumerate(line[:-1].split(",")[:-1])
        ]
        for line in lines
    ]
    f.close()

    return data


print("使用ALS算法")
model = ALS()
# 数据加载
X = load_movie_ratings("./ratings_small.csv")
# print(X)
# 运行max_iter次，k=聚类个数
model.fit(X, k=3, max_iter=2)
"""
X = np.array([[1,1,1], [1,2,1], [2,1,1], [2,3,1], [3,2,1], [3,3,1], [4,1,1], [4,2,1],
              [5,4,1], [5,5,1], [6,4,1], [6,6,1], [7,5,1], [7,6,1], [8,4,1], [8,5,1], [9,4,1], [9,5,1],
              [10,7,1], [10,8,1], [11,8,1], [11,9,1], [12,7,1], [12,9,1]])
# 运行max_iter次
model.fit(X, k=3, max_iter=20)
"""

print("对用户进行推荐")
# 对用户1-12进行评分预测 => TopN推荐
user_ids = range(1, 13)
# 对用户列表user_ids，进行Top-n推荐
predictions = model.predict(user_ids, n_items=2)
print(predictions)
for user_id, prediction in zip(user_ids, predictions):
    _prediction = [format_prediction(item_id, score) for item_id, score in prediction]
    print("用户ID:%d 推荐结果: %s" % (user_id, _prediction))
