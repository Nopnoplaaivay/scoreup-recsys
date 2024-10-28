import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 

class CF(object):
    """docstring for CF"""
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None

        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data['user_id'].to_numpy())) + 1 
        self.n_items = int(np.max(self.Y_data['cluster'].to_numpy())) + 1

    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)

    def normalize_Y(self):
        users = self.Y_data["user_id"] # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # row indices of rec_score done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data.iloc[ids, 1] 
            # and the corresponding ratings 
            rec_score = self.Y_data.iloc[ids, 2]
            # take mean
            m = np.mean(rec_score) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data.iloc[ids, 2] = rec_score - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar_data.iloc[:, 2],
            (self.Ybar_data.iloc[:, 1], self.Ybar_data.iloc[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        eps = 1e-6
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def __pred(self, u, i, normalized = 1):
        """ 
        predict the rec_score of user u for item i (normalized)
        if you need the un
        """
        ids = np.where(self.Y_data.iloc[:, 1] == i)[0].astype(np.int32)
        print(f"ids: {ids}")
        users_done_i = (self.Y_data.iloc[ids, 0]).astype(np.int32)
        print(f"users_done_i: {users_done_i}")
        # Step 3: find similarity btw the current user and others 
        # who already done i
        sim = self.S[u, users_done_i]
        print(f"sim: {sim}")
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:] 
        print(f"a: {a}")

        # and the corresponding similarity levels
        nearest_s = sim[a]
        print(f"nearest_s: {nearest_s}")

        # How did each of 'near' users done item i
        r = self.Ybar[i, users_done_i[a]]
        print(f"r: {r}")

        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized = 1):
        if self.uuCF: 
            return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)
    
    def recommend(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which 
        have not been done by u yet. 
        """
        ids = np.where(self.Y_data.iloc[:, 0] == u)[0]
        items_done_by_u = self.Y_data.iloc[ids, 1].tolist()              
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_done_by_u:
                # rec_score = self.__pred(u, i)
                # test print
                recommended_items.append(i)
                # if rec_score > 0: 
                #     recommended_items.append(i)
        
        return recommended_items 