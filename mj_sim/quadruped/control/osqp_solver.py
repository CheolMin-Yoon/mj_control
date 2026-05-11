import scipy.sparse as sp
import osqp


class OSQP_Solver:
    """min ½ xᵀ H x + qᵀ x   s.t.  l ≤ A x ≤ u

    setup() 1회 → CSC sparsity pattern 고정 + KKT factorize.
    update() 매 step → CSC data array만 in-place 갱신, factorization 재사용.

    NOTE: setup 이후 H, A의 sparsity 패턴이 바뀌면 안 됨 (nonzero 위치 불변 가정).
          MPC contact-mask로 컬럼이 0이 되는 정도는 nonzero 패턴 유지 — 그대로 OK.
    """

    def __init__(self, n, m, eps=1e-4, polish=False):
        self._qp = osqp.OSQP()
        self._n, self._m = n, m
        self._setup_done = False
        self._opts = dict(eps_abs=eps, eps_rel=eps, max_iter=500,
                          polish=polish, verbose=False, warm_starting=True)

    def setup(self, H, q, A, l, u):
        self._P_sp = sp.triu(sp.csc_matrix(H), format='csc')
        self._A_sp = sp.csc_matrix(A)

        P_coo = self._P_sp.tocoo()
        A_coo = self._A_sp.tocoo()
        self._P_rows, self._P_cols = P_coo.row, P_coo.col
        self._A_rows, self._A_cols = A_coo.row, A_coo.col

        self._qp.setup(self._P_sp, q, self._A_sp, l, u, **self._opts)
        self._setup_done = True

    def update(self, H=None, q=None, A=None, l=None, u=None):
        kw = {}
        if H is not None:
            self._P_sp.data[:] = H[self._P_rows, self._P_cols]
            kw['Px'] = self._P_sp.data
        if A is not None:
            self._A_sp.data[:] = A[self._A_rows, self._A_cols]
            kw['Ax'] = self._A_sp.data
        if q is not None: kw['q'] = q
        if l is not None: kw['l'] = l
        if u is not None: kw['u'] = u
        self._qp.update(**kw)

    def solve(self):
        res = self._qp.solve()
        if res.info.status == 'solved':
            return res.x
        print(f"[OSQP] solve failed: status={res.info.status} "
              f"| iter={res.info.iter} | prim_res={res.info.pri_res:.3e} | dual_res={res.info.dua_res:.3e}")
        return None
