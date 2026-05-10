import numpy as np
import scipy.sparse as sp
import osqp


class OSQP_Solvers:
    """min ½ xᵀ H x + qᵀ x   s.t.  l ≤ A x ≤ u

    setup() 한 번 → update() 매 step (factorization 재사용 + warm-start).
    """

    def __init__(self, n, m, eps=1e-4, polish=False):
        self._qp = osqp.OSQP()
        self._n, self._m = n, m
        self._setup_done = False
        self._opts = dict(eps_abs=eps, eps_rel=eps, max_iter=500,
                          polish=polish, verbose=False, warm_starting=True)

    def setup(self, H, q, A, l, u):
        P = sp.triu(sp.csc_matrix(H), format='csc')
        A_sp = sp.csc_matrix(A)
        self._qp.setup(P, q, A_sp, l, u, **self._opts)
        self._setup_done = True

    def update(self, H=None, q=None, A=None, l=None, u=None):
        kw = {}
        if H is not None: kw['Px'] = sp.triu(sp.csc_matrix(H), format='csc').data
        if q is not None: kw['q']  = q
        if A is not None: kw['Ax'] = sp.csc_matrix(A).data
        if l is not None: kw['l']  = l
        if u is not None: kw['u']  = u
        self._qp.update(**kw)

    def solve(self):
        res = self._qp.solve()
        return res.x if res.info.status == 'solved' else None
