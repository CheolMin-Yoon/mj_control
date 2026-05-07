import numpy as np
import pinocchio as pin
import proxsuite
from scipy.linalg import lstsq


class WholeBodyTorqueGenerator:
    """floating-base EOM + contact constraint → tau (na,).

    G z = f  →  augmented LS:  [ G   ] z = [ f ]
                                [ √W  ]     [ 0 ]

    G = [M(q)   -S^T ],  f = [ -nle + J_c^T F_hat ]
        [J_c(q)   0  ]       [ -Jdot_qdot          ]

    z = [ddq(nv), tau(na)].  SVD(gelsd)로 직접 풀이 (G^TG 정규방정식보다 안전).
    """

    def __init__(self, nv: int, na: int, nc: int,
                 contact_keys: list,
                 w_ddq: float = 1e-4, w_tau: float = 1e-3):
        self.nv, self.na, self.nc = nv, na, nc
        self.contact_keys = list(contact_keys)
        self.n_z = nv + na

        self.G = np.zeros((nv + nc, self.n_z))

        # -S^T: floating base(6) 행 = 0, actuated(na) 행 = -I (고정)
        self.G[6:nv, nv:] = -np.eye(na)

        # Tikhonov regularization
        sqrtW = np.zeros((self.n_z, self.n_z))
        np.fill_diagonal(sqrtW[:nv, :nv], np.sqrt(w_ddq))
        np.fill_diagonal(sqrtW[nv:, nv:], np.sqrt(w_tau))

        # augmented system 사전 할당: A = [G; √W], b = [f; 0]
        self._A = np.zeros((nv + nc + self.n_z, self.n_z))
        self._A[nv + nc:, :] = sqrtW
        self._b = np.zeros(nv + nc + self.n_z)

        self.J_c        = np.zeros((nc, nv))
        self.Jdot_qdot  = np.zeros(nc)


    def compute(self, pw, F_hat: np.ndarray,
                contact_mask=None) -> np.ndarray:
        """
        pw           : Pinocchio_Wrapper (update_model 완료 상태)
        F_hat        : (nc,) 최적 접촉력
        contact_mask : (n_feet,) int, 1=stance 0=swing. None이면 전부 stance 가정.
        returns      : tau (na,)
        """
        n_feet = len(self.contact_keys)
        dof_per_foot = self.nc // n_feet

        # contact Jacobian + Jdot*qdot 갱신
        for i, key in enumerate(self.contact_keys):
            rows = slice(i * dof_per_foot, (i + 1) * dof_per_foot)
            self.J_c[rows]       = pw.get_J_world(key)
            self.Jdot_qdot[rows] = pw.get_Jdot_dq_world(key)

        # swing 발 행 무효화 (접촉 제약 제외)
        if contact_mask is not None:
            for i, c in enumerate(contact_mask):
                if c == 0:
                    rows = slice(i * dof_per_foot, (i + 1) * dof_per_foot)
                    self.J_c[rows]       = 0.0
                    self.Jdot_qdot[rows] = 0.0

        M, nle = pw.M, pw.nle

        # G 블록 갱신
        self.G[:self.nv, :self.nv] = M
        self.G[self.nv:, :self.nv] = self.J_c

        # f 갱신
        self._b[:self.nv]               = -nle + self.J_c.T @ F_hat
        self._b[self.nv:self.nv+self.nc] = -self.Jdot_qdot
        # _b[nv+nc:] = 0 (항상 0 — 사전 할당 시 초기화됨)

        # augmented LS
        self._A[:self.nv+self.nc, :] = self.G
        z, *_ = lstsq(self._A, self._b, lapack_driver='gelsd', check_finite=False)

        return z[self.nv:]   # tau (na,)


class CoMDynamics:
    """선/각 운동량 동역학 → 접촉력 QP의 K, u.

    K F = u,  F (nc_f,) = [f_L(3), τ_L(3), f_R(3), τ_R(3)]  (contact_keys 순)

    [ D1 ] F = [ m·ddc_des + F_g ]
    [ D2 ]     [ dL               ]
    """

    def __init__(self, mass: float, gravity: float = 9.81):
        self.m = mass
        self.g = gravity
        self.K = np.zeros((6, 12))
        self.u = np.zeros(6)

        # D1 = [I 0 I 0]  (선형 운동량, 고정 — contact_keys 첫째=L, 둘째=R)
        self._D1 = np.zeros((3, 12))
        self._D1[:, 0:3] = np.eye(3)   # f_L
        self._D1[:, 6:9] = np.eye(3)   # f_R


    def update(self, pw, des_acc: np.ndarray, dL: np.ndarray = None):
        """
        pw      : Pinocchio_Wrapper (update_model 완료)
        des_acc : (3,) desired CoM acceleration [world]
        dL      : (3,) rate of angular momentum (default 0 — LIPM 가정)
        """
        if dL is None:
            dL = np.zeros(3)

        com_pos = pw.com_pos_world
        r_L = pw.oM_Lfoot.translation - com_pos
        r_R = pw.oM_Rfoot.translation - com_pos

        # K = [ I    0   I    0  ]
        #     [ [r_L]× I [r_R]× I ]
        self.K[:3] = self._D1
        self.K[3:6, 0:3]  = pin.skew(r_L)
        self.K[3:6, 3:6]  = np.eye(3)
        self.K[3:6, 6:9]  = pin.skew(r_R)
        self.K[3:6, 9:12] = np.eye(3)

        self.u[0]  = self.m * des_acc[0]
        self.u[1]  = self.m * des_acc[1]
        self.u[2]  = self.m * des_acc[2] + self.m * self.g
        self.u[3:] = dL


class ForceOptimizerProx:
    """ProxQP 기반 접촉력 최적화.

    F (12,) = [f_L(3), τ_L(3), f_R(3), τ_R(3)]  (contact_keys 순: L first)

    warm-start 유지: __init__ 에서 메모리 사전 할당, solve() 호출마다 H/g만 교체.
    """

    N_VARS = 12
    N_EQ   = 0
    N_INEQ = 10   # 마찰원뿔 5행 × 2발

    def __init__(self, mu: float = 0.5, fz_max: float = 500.0):
        self.mu = mu
        self.fz_max = fz_max
        self.opt_F = np.zeros(self.N_VARS)

        self._A, self._l, self._u = self._build_constraints()

        self._qp = proxsuite.proxqp.dense.QP(self.N_VARS, self.N_EQ, self.N_INEQ)
        self._qp.settings.eps_abs      = 1e-4
        self._qp.settings.eps_rel      = 1e-4
        self._qp.settings.max_iter     = 1000
        self._qp.settings.verbose      = False
        self._qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT

        H0 = np.eye(self.N_VARS)
        g0 = np.zeros(self.N_VARS)
        self._qp.init(H0, g0, None, None, self._A, self._l, self._u)
        self._initialized = False

        # 사전 할당 버퍼
        self._W_reg = 1e-4 * np.eye(self.N_VARS)
        self._H_buf = np.empty((self.N_VARS, self.N_VARS))
        self._g_buf = np.empty(self.N_VARS)


    def _build_constraints(self):
        mu_eff = self.mu / np.sqrt(2.0)
        INF = 1e9
        A3 = np.array([[ 1,  0, -mu_eff],
                       [-1,  0, -mu_eff],
                       [ 0,  1, -mu_eff],
                       [ 0, -1, -mu_eff],
                       [ 0,  0,  1     ]], dtype=np.float64)
        l3 = np.array([-INF, -INF, -INF, -INF, 0.0])
        u3 = np.array([ 0.0,  0.0,  0.0,  0.0, self.fz_max])

        A6 = np.zeros((5, 6))
        A6[:, :3] = A3

        A = np.zeros((self.N_INEQ, self.N_VARS))
        A[:5,  0:6]  = A6   # L_foot
        A[5:, 6:12]  = A6   # R_foot
        return A, np.concatenate([l3, l3]), np.concatenate([u3, u3])


    def solve(self, K: np.ndarray, u_vec: np.ndarray,
              W_reg: np.ndarray = None) -> np.ndarray:
        """
        K     : (6, 12)
        u_vec : (6,)
        W_reg : (12, 12) optional. None이면 기본 _W_reg 사용.
        returns opt_F (12,)
        """
        Wreg = self._W_reg if W_reg is None else W_reg

        np.dot(K.T, K, out=self._H_buf)
        self._H_buf += Wreg
        self._H_buf *= 2.0

        np.dot(K.T, u_vec, out=self._g_buf)
        self._g_buf *= -2.0

        if not self._initialized:
            self._qp.init(self._H_buf, self._g_buf,
                          None, None, self._A, self._l, self._u)
            self._initialized = True
        else:
            self._qp.update(H=self._H_buf, g=self._g_buf)

        self._qp.solve()

        if self._qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
            self.opt_F = np.array(self._qp.results.x)

        return self.opt_F


class WholeBodyController:
    """SRBD balance 제어 파이프라인.

    BalanceTask (PD) → CoMDynamics → ForceOptimizer (QP) → WholeBodyTorqueGenerator

    외부에서 pw.update_model(q, dq) 호출 후 compute() 사용.
    contact_keys 순서는 LEG_KEYS = ["L_foot", "R_foot"] 와 일치.
    """

    def __init__(self, pw,
                 kp_com: float = 1.0,
                 kd_com: float = 0.2,
                 mu: float = 0.5,
                 nc: int = 12,
                 contact_keys=None,
                 w_ddq: float = 1e-4,
                 w_tau: float = 1e-3):
        self.pw           = pw
        self.kp_com       = kp_com
        self.kd_com       = kd_com
        self.contact_keys = list(contact_keys) if contact_keys else ["L_foot", "R_foot"]

        self.torque_gen = WholeBodyTorqueGenerator(pw.nv, pw.na, nc,
                                                   contact_keys=self.contact_keys,
                                                   w_ddq=w_ddq, w_tau=w_tau)
        self.com_dyn    = CoMDynamics(pw.mass)
        self.force_opt  = ForceOptimizerProx(mu=mu)


    def compute(self, com_des: np.ndarray, com_dot_des: np.ndarray,
                contact_mask=None, dL_des: np.ndarray = None,
                ddc_feedforward: np.ndarray = None) -> np.ndarray:
        """
        com_des         : (3,) desired CoM position [world]
        com_dot_des     : (3,) desired CoM velocity [world]
        contact_mask    : (n_feet,) int, 1=stance 0=swing (LEG_KEYS 순)
        ddc_feedforward : (3,) 외부 (예: MPC) 가 plan 한 desired CoM acceleration —
                          PD 출력에 더해진다 (replace 아님).  PD 가 안정성/고대역 피드백을
                          담당하고, MPC 가 horizon-aware 명령을 feedforward 로 추가.
        returns         : tau (na,)
        """
        pw = self.pw

        # BalanceTask: PD → ddc_des
        ddc_des = (self.kp_com * (com_des     - pw.com_pos_world) +
                   self.kd_com * (com_dot_des - pw.com_vel_world))
        if ddc_feedforward is not None:
            ddc_des = ddc_des + ddc_feedforward

        # CoMDynamics: K, u
        self.com_dyn.update(pw, ddc_des, dL=dL_des)

        # ForceOptimizer: swing 발 힘을 0으로 유도하는 대형 패널티
        if contact_mask is not None:
            W_swing = self.force_opt._W_reg.copy()
            for i, c in enumerate(contact_mask):
                if c == 0:
                    W_swing[i*6:(i+1)*6, i*6:(i+1)*6] = 1e4 * np.eye(6)
            F_hat = self.force_opt.solve(self.com_dyn.K, self.com_dyn.u, W_reg=W_swing)
        else:
            F_hat = self.force_opt.solve(self.com_dyn.K, self.com_dyn.u)

        # WholeBodyTorqueGenerator
        tau = self.torque_gen.compute(pw, F_hat, contact_mask)

        return tau


    def get_desired_forces(self) -> dict:
        """→ {key: f(3,)} 목표 접촉력 [N], world frame."""
        F = self.force_opt.opt_F  # (12,)
        return {
            self.contact_keys[0]: F[0:3].copy(),
            self.contact_keys[1]: F[6:9].copy(),
        }
