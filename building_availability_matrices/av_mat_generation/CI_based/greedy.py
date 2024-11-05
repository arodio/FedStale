import numpy as np
import copy

class GreedyProblem(object):
    """
    The greedy approximation iteratively adds the element v that maximizes 
    the benefit cost ratio among all elements still affordable with the 
    remaining budget.
    The current problem does not just have cardinality constraints, it has
    Knapsack constraints (nonnegative modular constraint).
    Reference: KRAUSE, Andreas et GOLOVIN, Daniel. Submodular function maximization. 
    Tractability, 2014, vol. 3, no 71-104, p. 3.
    """
    def __init__(self, GHG_matrix, alpha_f, w):
        self.GHG_mat = GHG_matrix.to_numpy()
        self.one_m_GHG_w = (np.max(self.GHG_mat) - self.GHG_mat)@np.diag(w)
        # self.one_m_GHG_w = (1 - self.GHG_mat/np.max(self.GHG_mat))@np.diag(w) # test
        # self.one_m_GHG_w = (np.diag(np.max(self.GHG_mat, axis=1))@np.ones((n_clients, n_rounds)) - self.GHG_mat)@np.diag(w) # test
        self.GHG_w = self.GHG_mat@np.diag(w) # test
        self.alpha_f = alpha_f
        self.n_clients = GHG_matrix.shape[0]
        self.n_rounds = GHG_matrix.shape[1]
        self.initialize_A()

    def initialize_A(self):
        self.A = np.zeros(self.GHG_mat.shape)

    def obj_func(self, A):
        # remove the minus here if you want to maximize instead of minimizing (argmin -> argmax):
        return np.sum(np.power(np.sum(np.multiply(self.one_m_GHG_w, A), axis=1), self.alpha_f))
        # return np.sum(np.power(np.sum(np.multiply(self.GHG_w, A), axis=1), self.alpha_f)) # test
        # return np.sum(np.power(np.sum(np.multiply(self.GHG_w, np.ones(A.shape) - A), axis=1), self.alpha_f)) # test
    
        # return -np.sum(np.sum(np.multiply(self.one_m_GHG_w, A), axis=1))

    def diff(self, i, j):
        A_new = copy.copy(self.A)
        A_new[i, j] = 1
        # return self.obj_func(A_new) - self.obj_func(self.A)
        return (self.obj_func(A_new) - self.obj_func(self.A))/self.GHG_mat[i, j] # test
        # return self.obj_func(A_new) # something else we can optimize
    
    def update_A(self, i_star, j_star):
        self.A[i_star, j_star] = 1

    def greedy_optimization(self, carbon_budget):
        self.initialize_A()
        indexes = [(i, j) for i in range(self.n_clients) for j in range(self.n_rounds)]
        mask = np.array([0 for i in range(self.n_clients) for j in range(self.n_rounds)])
        G = carbon_budget

        while True:
            values = np.array([self.diff(*idx) for idx in indexes])

            shuf_order = np.arange(len(values))
            shuf_values = values[shuf_order]
            shuf_mask = mask[shuf_order]
            # unshuf_order = np.zeros(len(values), dtype=np.int8)
            # unshuf_order[shuf_order] = np.arange(len(values))

            masked_arr = np.ma.masked_array(shuf_values, shuf_mask)
            # shuf_res = masked_arr.argmin()
            shuf_res = masked_arr.argmax()

            res = shuf_order[shuf_res]

            i_star = res//self.n_rounds
            j_star = res - i_star*self.n_rounds

            G -= self.GHG_mat[i_star, j_star]
            if G < 0:
                break

            self.update_A(i_star, j_star)
            mask[res] = 1
            # indexes.remove((i_star, j_star))