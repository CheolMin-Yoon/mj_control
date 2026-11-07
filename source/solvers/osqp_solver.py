"""OSQP 반복 QP, setup 한 번 뒤 update·solve로 factorization 재사용과 warm start"""

import numpy as np
import osqp
import scipy.sparse as sp


class OSQPSolver:
    """min ½xᵀHx + cᵀx  s.t.  l ≤ Ax ≤ u, H sparsity pattern은 setup 때 고정

    인자:
        eps: 절대·상대 허용오차
        max_iter: ADMM 최대 반복
        polishing: OSQP solution polishing
    """

    def __init__(self, eps=1e-4, max_iter=500, polishing=False):
        self._qp = osqp.OSQP()
        self._settings = dict(
            eps_abs=eps,
            eps_rel=eps,
            max_iter=max_iter,
            polishing=polishing,
            verbose=False,
            warm_starting=True,
        )

    def setup(self, H, linear_cost, A, lower, upper):
        """H의 상삼각 pattern 기록, 이후 update의 값 배열 길이 고정"""
        P = sp.triu(sp.csc_matrix(H), format="csc")
        P_coo = P.tocoo()
        self._P_rows, self._P_cols = P_coo.row, P_coo.col
        self._qp.setup(P, linear_cost, sp.csc_matrix(A), lower, upper, **self._settings)

    def update(self, H, linear_cost, lower, upper):
        # 0 원소가 생겨도 setup pattern 순서·길이 유지 → csc 재변환 없이 index로 추출
        Px = np.asarray(H)[self._P_rows, self._P_cols]
        self._qp.update(Px=Px, q=linear_cost, l=lower, u=upper)

    def solve(self):
        """해 x, 미해결이면 None"""
        result = self._qp.solve(raise_error=False)
        return result.x if result.info.status_val == osqp.SolverStatus.OSQP_SOLVED else None
