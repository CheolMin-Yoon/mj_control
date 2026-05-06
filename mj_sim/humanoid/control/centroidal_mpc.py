import time
import numpy as np
import scipy.sparse as sp
import casadi as ca


# OSQP 옵션 (go2-convex-mpc 동일 세팅)
OSQP_OPTS = {
    'warm_start_primal': True,
    'warm_start_dual':   True,
    'osqp': {
        'eps_abs':              1e-4,
        'eps_rel':              1e-4,
        'max_iter':             1000,
        'polish':               False,
        'verbose':              False,
        'adaptive_rho':         True,
        'adaptive_rho_interval': 25,
        'check_termination':    10,
        'scaling':              5,
        'scaled_termination':   True,
    },
}


class CentroidalMPC:
    """SRBD horizon-N convex MPC (n_legs 일반화).

    상태 컨벤션 — TrajOptimizer 와 동일:
      x (13,) = [Θ(3), p(3), ω(3), v(3), 1]
        Θ : base 오일러각 (roll/pitch/yaw)
        p : CoM 위치 (world)
        ω : base 각속도 (body)
        v : CoM 선속도 (world)
        1 : 증대 상태 — 중력항을 A_c[9:12,12]=[0,0,-g]로 흡수, x_0[12]=1 필수

    입력:
      u (3·n_legs,) — leg별 3D contact force (world frame, legs 순서대로)

    Decision:
      z = [x_1, x_2, ..., x_N, u_0, u_1, ..., u_{N-1}]   (N*(NX+NU),)
      x_0는 측정값, decision에서 제외

    Equality (이산 동역학):
      x_{k+1} = A_d @ x_k + B_d_h[k] @ u_k     k=0..N-1

    Inequality (마찰원뿔, leg×step별 4면):
      [+1, 0, -μ; -1, 0, -μ; 0, +1, -μ; 0, -1, -μ] @ f_leg <= 0   (stance)
                                                   <= +inf (swing — box로 force=0 강제)

    Box:
      stance leg: f_z ∈ [fz_min, fz_max], f_x/f_y unbounded
      swing  leg: f_x = f_y = f_z = 0
      x[12] 는 dynamics가 1로 propagate (x_0[12]=1 보장 시)

    Cost:
      Σ ‖x_k − x_ref_k‖²_Q + Σ ‖u_k‖²_R     (Q[12,12]=0 권장)
    """

    NX = 13   # TrajOptimizer 컨벤션 고정

    def __init__(self, traj_opt, N: int, dt: float,
                 Q_diag, R_diag,
                 mu: float = 0.5,
                 fz_min: float = 10.0,
                 fz_max: float = 500.0):
        self.traj_opt = traj_opt
        self.N        = int(N)
        self.dt       = float(dt)
        self.n_legs   = len(traj_opt.legs)
        self.NU       = 3 * self.n_legs
        self.mu       = float(mu)
        self.fz_min   = float(fz_min)
        self.fz_max   = float(fz_max)

        Q = np.asarray(Q_diag, dtype=float)
        R = np.asarray(R_diag, dtype=float)
        assert Q.shape == (self.NX,), f"Q_diag shape {Q.shape} != ({self.NX},)"
        assert R.shape == (self.NU,), f"R_diag shape {R.shape} != ({self.NU},)"
        self.Q = np.diag(Q)
        self.R = np.diag(R)

        self.nvars = self.N * self.NX + self.N * self.NU

        # 1) Hessian (constant — Q, R diagonal에서 결정)
        self.H_const = self._build_hessian()

        # 2) Friction inequality 행렬 (constant — bound만 step별 토글)
        self.A_ineq_static = self._build_friction_matrix()

        # 3) Solver init (sparsity 고정용 placeholder)
        self._init_solver()

        # warm start state
        self.x_prev     = None
        self.lam_x_prev = None
        self.lam_a_prev = None

        # diagnostic
        self.update_time = 0.0
        self.solve_time  = 0.0
        self.last_status = None


    # -------- public --------
    def solve(self, x0: np.ndarray, x_ref_horizon: np.ndarray,
              contact_table: np.ndarray, verbose: bool = False) -> np.ndarray:
        """1회 MPC 풀이 → horizon 전체 contact force 시퀀스 반환.

        x0             : (13,)         현재 augmented state (x0[12]=1 필수)
        x_ref_horizon  : (N, 13)       reference trajectory
        contact_table  : (n_legs, N)   int 1=stance 0=swing

        returns: F_seq (N, NU)         world frame contact force
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        assert x0.shape == (self.NX,)
        assert x_ref_horizon.shape == (self.N, self.NX)
        assert contact_table.shape == (self.n_legs, self.N)

        t0 = time.perf_counter()

        # 1) Dynamics 갱신 (TrajOptimizer가 A_d_h, B_d_h 캐싱)
        self.traj_opt.update_dynamics_horizon(contact_table, self.dt)
        Ad = self.traj_opt.A_d_h
        Bd = self.traj_opt.B_d_h   # (N, 13, NU)

        # 2) Constraint matrix + bounds + linear cost
        A_total, lba, uba = self._update_constraints(Ad, Bd, x0, contact_table)
        lbx, ubx          = self._compute_box_bounds(contact_table)
        g                 = self._build_linear_cost(x_ref_horizon)

        t1 = time.perf_counter()

        # 3) QP solve (warm start 지원)
        args = {'h': self.H_const, 'g': g, 'a': A_total,
                'lba': lba, 'uba': uba, 'lbx': lbx, 'ubx': ubx}
        if self.x_prev is not None:
            args['x0']     = self.x_prev
            args['lam_x0'] = self.lam_x_prev
            args['lam_a0'] = self.lam_a_prev

        sol = self.solver(**args)
        t2 = time.perf_counter()

        # 4) Stash for warm start next call
        self.x_prev     = sol['x']
        self.lam_x_prev = sol['lam_x']
        self.lam_a_prev = sol['lam_a']

        self.update_time = (t1 - t0) * 1e3
        self.solve_time  = (t2 - t1) * 1e3
        self.last_status = self.solver.stats().get('return_status')

        # 5) F_seq 추출
        z = np.asarray(sol['x']).flatten()
        F_seq = z[self.N * self.NX:].reshape(self.N, self.NU)

        if verbose:
            print(f"[MPC] update {self.update_time:.2f}ms | solve {self.solve_time:.2f}ms"
                  f" | status {self.last_status}")

        return F_seq


    # -------- internals --------
    def _build_hessian(self):
        """블록 대각: 2Q (N블록), 2R (N블록). triplet 으로 sparse 작성."""
        rows, cols, vals = [], [], []
        # state cost
        for k in range(self.N):
            base = k * self.NX
            for i in range(self.NX):
                if self.Q[i, i] != 0.0:
                    rows.append(base + i); cols.append(base + i)
                    vals.append(2.0 * self.Q[i, i])
        # input cost
        for k in range(self.N):
            base = self.N * self.NX + k * self.NU
            for i in range(self.NU):
                if self.R[i, i] != 0.0:
                    rows.append(base + i); cols.append(base + i)
                    vals.append(2.0 * self.R[i, i])
        H = ca.DM.triplet(rows, cols, ca.DM(vals), self.nvars, self.nvars)
        return H


    def _build_friction_matrix(self):
        """Friction pyramid 4면 — leg×step 전체."""
        n_rows = 4 * self.n_legs * self.N
        rows, cols, vals = [], [], []
        baseU = self.N * self.NX
        r0 = 0
        for k in range(self.N):
            uk0 = baseU + k * self.NU
            for leg in range(self.n_legs):
                fx = uk0 + 3 * leg
                fy = uk0 + 3 * leg + 1
                fz = uk0 + 3 * leg + 2
                # +x face: fx - μ·fz <= 0
                rows.extend([r0, r0]); cols.extend([fx, fz]); vals.extend([1.0, -self.mu]); r0 += 1
                # -x face
                rows.extend([r0, r0]); cols.extend([fx, fz]); vals.extend([-1.0, -self.mu]); r0 += 1
                # +y face
                rows.extend([r0, r0]); cols.extend([fy, fz]); vals.extend([1.0, -self.mu]); r0 += 1
                # -y face
                rows.extend([r0, r0]); cols.extend([fy, fz]); vals.extend([-1.0, -self.mu]); r0 += 1
        A_sp = sp.csc_matrix((vals, (rows, cols)), shape=(n_rows, self.nvars))
        return self._scipy_to_casadi(A_sp)


    def _build_dynamics_block(self, Ad, Bd):
        """Dynamics equality 행렬 [I_block - A_block_lower_diag, -B_block_diag] 조립.

          x_1 - A_d x_0 - B_0 u_0 = 0 → 첫 행: x_1 - B_0 u_0 = A_d x_0  (RHS로 이동)
          x_{k+1} - A_d x_k - B_k u_k = 0  for k>=1

        반환: (N*NX, N*(NX+NU)) sparse matrix
        """
        Ad_dm = ca.DM(Ad)

        # State block: I (대각) + S @ diag(-A_d, ..., -A_d) (S는 lower-shift)
        I_block  = ca.DM.eye(self.N * self.NX)
        ones_Nm1 = np.ones(max(self.N - 1, 0))
        S_scipy  = sp.kron(sp.diags([ones_Nm1], [-1]), sp.eye(self.NX))
        S_block  = self._scipy_to_casadi(S_scipy)

        big_minus_Ad = ca.diagcat(*([-Ad_dm] * self.N))
        state_block  = I_block + S_block @ big_minus_Ad

        # Input block: -B_k 가 대각 블록
        big_minus_Bd_list = [-ca.DM(Bd[k]) for k in range(self.N)]
        input_block = ca.diagcat(*big_minus_Bd_list)

        return ca.horzcat(state_block, input_block)


    def _update_constraints(self, Ad, Bd, x0, contact_table):
        """[A_eq; A_ineq] 와 lba/uba 조립."""
        A_eq = self._build_dynamics_block(Ad, Bd)

        # Eq RHS: 첫 블록 = A_d @ x_0, 나머지 = 0
        x0_dm     = ca.DM(x0.reshape(-1, 1))
        Ad_dm     = ca.DM(Ad)
        beq_first = Ad_dm @ x0_dm
        if self.N > 1:
            beq_rest = ca.DM.zeros((self.N - 1) * self.NX, 1)
            beq      = ca.vertcat(beq_first, beq_rest)
        else:
            beq = beq_first

        # Ineq bounds — stance면 활성(<=0), swing이면 비활성(<=+inf)
        n_ineq    = 4 * self.n_legs * self.N
        l_ineq    = -ca.inf * ca.DM.ones(n_ineq, 1)
        u_ineq_np = np.full(n_ineq, np.inf)
        idx = 0
        for k in range(self.N):
            for leg in range(self.n_legs):
                if contact_table[leg, k] == 1:
                    u_ineq_np[idx:idx + 4] = 0.0
                idx += 4
        u_ineq = ca.DM(u_ineq_np)

        A_total = ca.vertcat(A_eq, self.A_ineq_static)
        lba     = ca.vertcat(beq, l_ineq)
        uba     = ca.vertcat(beq, u_ineq)
        return A_total, lba, uba


    def _compute_box_bounds(self, contact_table):
        """State 자유, Input은 stance/swing별 box."""
        nvars  = self.nvars
        lbx_np = np.full(nvars, -np.inf)
        ubx_np = np.full(nvars,  np.inf)
        baseU  = self.N * self.NX
        for k in range(self.N):
            for leg in range(self.n_legs):
                base = baseU + k * self.NU + 3 * leg
                if contact_table[leg, k] == 0:
                    lbx_np[base:base + 3] = 0.0
                    ubx_np[base:base + 3] = 0.0
                else:
                    lbx_np[base + 2] = self.fz_min
                    ubx_np[base + 2] = self.fz_max
        return ca.DM(lbx_np.reshape(-1, 1)), ca.DM(ubx_np.reshape(-1, 1))


    def _build_linear_cost(self, x_ref_horizon):
        """g = [-2 Q x_ref_1; ...; -2 Q x_ref_N; 0; ...; 0]"""
        Q_dm     = ca.DM(self.Q)
        x_ref_dm = ca.DM(x_ref_horizon.T)            # (NX, N)
        gx_mat   = -2.0 * (Q_dm @ x_ref_dm)
        g_x      = ca.vec(gx_mat)
        g_u      = ca.DM.zeros(self.N * self.NU, 1)
        return ca.vertcat(g_x, g_u)


    def _init_solver(self):
        """Sparsity 고정용 placeholder dynamics로 conic solver 1회 생성."""
        Ad_zero = np.eye(self.NX)
        Bd_full = np.ones((self.N, self.NX, self.NU))   # 모든 entry를 structural nonzero로
        A_eq    = self._build_dynamics_block(Ad_zero, Bd_full)
        A_total = ca.vertcat(A_eq, self.A_ineq_static)

        self.H_sp = self.H_const.sparsity()
        self.A_sp = A_total.sparsity()

        qp = {'h': self.H_sp, 'a': self.A_sp}
        self.solver = ca.conic('mpc_solver', 'osqp', qp, OSQP_OPTS)


    @staticmethod
    def _scipy_to_casadi(M):
        M = M.tocsc()
        return ca.DM(ca.Sparsity(M.shape[0], M.shape[1], M.indptr, M.indices), M.data)
