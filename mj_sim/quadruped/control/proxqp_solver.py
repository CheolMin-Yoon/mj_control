from proxsuite import proxqp
qp_dense = proxqp.dense


class ProxQP_Solver:
    """min ½ xᵀ H x + qᵀ x   s.t.  l ≤ A x ≤ u  (dense backend)

    proxsuite.proxqp.dense — Eigen 기반, alloc 없는 update + warm-start.
    OSQP_Solver와 동일 인터페이스 — sparse 변환·CSC indexing 자체가 없음.

    NOTE: equality는 안 씀 (b=None). inequality C=A로 던짐.
    """

    def __init__(self, n, m, eps=1e-3, polish=True):
        self._n, self._m = n, m
        self._qp = qp_dense.QP(n, 0, m)
        self._qp.settings.eps_abs = eps
        self._qp.settings.eps_rel = eps
        # iter=1에서 false-positive PRIMAL_INFEASIBLE 종료 방지
        self._qp.settings.eps_primal_inf = 1e-10
        self._qp.settings.eps_dual_inf   = 1e-10
        self._qp.settings.verbose = False
        self._setup_done = False

    def setup(self, H, q, A, l, u):
        self._qp.init(H=H, g=q, A=None, b=None, C=A, l=l, u=u)
        self._setup_done = True

    def update(self, H=None, q=None, A=None, l=None, u=None):
        self._qp.update(H=H, g=q, C=A, l=l, u=u)

    def solve(self):
        self._qp.solve()
        status = self._qp.results.info.status
        if str(status).endswith('PROXQP_SOLVED'):
            return self._qp.results.x
        info = self._qp.results.info
        print(f"[ProxQP] solve failed: status={status} "
              f"| iter={info.iter} | prim_res={info.pri_res:.3e} | dual_res={info.dua_res:.3e}")
        return None
