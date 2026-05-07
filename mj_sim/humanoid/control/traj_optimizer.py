import numpy as np
import pinocchio as pin

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Pinocchio_Wrapper

LEG_KEYS_DEFAULT = ("L_foot", "R_foot")


class TrajOptimizer:
    """SRBD 연속/이산 동역학 + CoM 참조 명령 인터페이스.

    상태 x (13,) = [Θ(3), p(3), ω(3), v(3), 1]
      Θ : base 오일러각 (roll/pitch/yaw), body-z 기준
      p : CoM 위치 (world)
      ω : base 각속도 (world)   — go2-convex-mpc 컨벤션, B_c도 world frame moment arm
      v : CoM 선속도 (world)
      1 : 증대 상태 (중력 상수항 흡수)

    입력 u (3·n_feet,) — stance 발 3D force (world), 발 순서는 legs 인자와 동일.
    """

    def __init__(self, wrapper: 'Pinocchio_Wrapper',
                 legs=LEG_KEYS_DEFAULT):
        self.wrapper = wrapper
        self.legs    = list(legs)
        n_f          = 3 * len(self.legs)

        # SRBD 관성 (neutral pose 캐싱 — 보행 중 크게 안 변함)
        self._I_inv = np.linalg.inv(wrapper.inertia_matrix)
        self._mass  = wrapper.mass

        # 연속 모델
        self.A_c = np.zeros((13, 13))
        self.A_c[3:6, 9:12] = np.eye(3)               # p_dot = v
        self.A_c[9:12, 12]  = np.array([0, 0, -9.81]) # v_dot += [0,0,-g]×x[12]

        self.B_c = np.zeros((13, n_f))

        # 이산 모델 (discretize 호출 후 채워짐)
        self.A_d = np.eye(13)
        self.B_d = np.zeros((13, n_f))

        # 참조 명령 (set_reference 로 갱신)
        self._cmd = np.zeros(5)  # [pos_x, pos_y, vel_x, vel_y, yaw_rate]


    def set_reference(self, pos_x=0.0, pos_y=0.0,
                      vel_x=0.0, vel_y=0.0, yaw_rate=0.0):
        """텔레옵/조이스틱 참조 명령 갱신."""
        self._cmd[:] = [pos_x, pos_y, vel_x, vel_y, yaw_rate]


    def update_dynamics(self, contact_mask):
        """현재 stance 발 모멘트 암으로 A_c, B_c 갱신. contact_mask: (n_legs,) int, 1=stance."""
        self.A_c[0:3, 6:9] = self.wrapper.R_z.T
        self.B_c[:] = 0.0
        for i, (leg, c) in enumerate(zip(self.legs, contact_mask)):
            if c:
                r   = self.wrapper.get_moment_arm_in_world(leg)
                col = 3 * i
                self.B_c[6:9, col:col+3]  = self._I_inv @ pin.skew(r)
                self.B_c[9:12, col:col+3] = np.eye(3) / self._mass


    def discretize(self, dt: float):
        """2차 Taylor 이산화: A_d ≈ I + A_c·dt + (A_c·dt)²/2."""
        I13   = np.eye(13)
        Ac_dt = self.A_c * dt
        self.A_d = I13 + Ac_dt + 0.5 * Ac_dt @ Ac_dt
        self.B_d = (I13 * dt + 0.5 * Ac_dt * dt) @ self.B_c


    def update_dynamics_horizon(self, contact_table: np.ndarray, dt: float):
        """Horizon-N MPC용 stacked SRBD 이산 모델.

        contact_table : (n_legs, N) int, 1=stance / 0=swing
        dt            : MPC step duration [s]

        결과:
          A_d_h : (13, 13)        — A_c yaw 회전만 갱신, step별 동일
          B_d_h : (N, 13, n_f)    — step별 stance 패턴 반영 B_c 이산화

        단순화: 모멘트 암 r_i를 horizon 전체에 대해 현재시점 wrapper 기준으로 사용.
        (정확한 미래 r은 ref trajectory에서 와야 하지만 horizon 짧으면 무시 가능)
        """
        n_legs = len(self.legs)
        N      = int(contact_table.shape[1])
        n_f    = 3 * n_legs

        # 현재 yaw 기준 A_c 갱신
        self.A_c[0:3, 6:9] = self.wrapper.R_z.T

        I13   = np.eye(13)
        Ac_dt = self.A_c * dt
        self.A_d_h = I13 + Ac_dt + 0.5 * Ac_dt @ Ac_dt

        # 모멘트 암 캐싱 (현재시점 1회만)
        r_legs = [self.wrapper.get_moment_arm_in_world(leg) for leg in self.legs]

        # ZOH 근사 이산화 행렬 (B_c → B_d)
        discretizer = I13 * dt + 0.5 * Ac_dt * dt

        self.B_d_h = np.zeros((N, 13, n_f))
        Bc_step    = np.zeros((13, n_f))
        for k in range(N):
            Bc_step.fill(0.0)
            for i, leg in enumerate(self.legs):
                if contact_table[i, k]:
                    col = 3 * i
                    Bc_step[6:9,  col:col+3] = self._I_inv @ pin.skew(r_legs[i])
                    Bc_step[9:12, col:col+3] = np.eye(3) / self._mass
            self.B_d_h[k] = discretizer @ Bc_step


    def compute_desired_command(self):
        """→ (pos_x_des, pos_y_des, vel_x_des, vel_y_des, yaw_rate_des)"""
        return tuple(self._cmd)
