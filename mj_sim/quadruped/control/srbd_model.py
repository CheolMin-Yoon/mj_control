"""
SRBD (Single Rigid Body Dynamics) 13D 모델.

상태 x (13,):  [rpy(3), p(3), omega(3), v(3), -g_aug(1)]
입력 u (12,):  [f_FL(3), f_FR(3), f_RL(3), f_RR(3)]   GRF in world frame

연속 시간:    A_c (13×13), B_c (13×12)
이산 시간:    A_d (13×13), B_d (13×12)   (2차 운동학 적분 기반)
"""

import numpy as np
import pinocchio as pin

from core import Pinocchio_Wrapper

GRAVITY = 9.81


class SRBDModel:
    def __init__(self, wrapper: Pinocchio_Wrapper, legs: list):
        self.wrapper = wrapper
        self.legs    = legs
        self.n_legs  = len(legs)

        # 연속 / 이산 행렬
        self.A_c = np.zeros((13, 13))
        self.B_c = np.zeros((13, 3 * self.n_legs))
        self.A_d = np.zeros((13, 13))
        self.B_d = np.zeros((13, 3 * self.n_legs))

        # static — m, I_inv_body 시간 불변
        self.m = self.wrapper.m
        self.I_inv_body = np.linalg.inv(self.wrapper.I_body)

        # A_c 상수 블록
        self.A_c[3:6, 9:12] = np.eye(3)                       # d(p)/dt = v
        self.A_c[9:12, 12]  = np.array([0, 0, -GRAVITY])      # d(v)/dt = g

    def update_continuous_dynamics(self):
        """A_c 변동 블록 + B_c 갱신 (R_yaw·foot 위치 의존)."""
        self.A_c[0:3, 6:9] = self.wrapper.R_yaw.T             # 각속도 → 오일러 각
        I_inv_world = self.wrapper.R_yaw @ self.I_inv_body @ self.wrapper.R_yaw.T

        for i, leg in enumerate(self.legs):
            r       = self.wrapper.compute_moment_arm_W(leg)
            r_skew  = pin.skew(r)
            c0, c1  = 3 * i, 3 * (i + 1)

            # 각가속도 파트: I_inv_world · [r_i]_x
            self.B_c[6:9, c0:c1] = I_inv_world @ r_skew
            # 선가속도 파트: 1/m · I_3
            self.B_c[9:12, c0:c1] = np.eye(3) / self.m

    def update_discrete_dynamics(self, dt: float):
        """2차 운동학 적분 기반 이산화."""
        self.A_d = np.eye(13)
        self.A_d[0:3, 6:9]  = self.wrapper.R_yaw.T * dt        # 자세 적분
        self.A_d[3:6, 9:12] = np.eye(3) * dt                   # 위치 적분

        # 중력 (2차 적분: pos += 0.5·g·dt², vel += g·dt)
        g_vec = np.array([0, 0, -GRAVITY])
        self.A_d[3:6,  12] = 0.5 * g_vec * (dt ** 2)
        self.A_d[9:12, 12] = g_vec * dt

        # B_d = B_c · dt (가속도 → 속도) + 0.5·B_c·dt² (가속도 → 위치)
        for i in range(self.n_legs):
            c0, c1 = 3 * i, 3 * (i + 1)

            # 속도 파트
            self.B_d[6:9,  c0:c1] = self.B_c[6:9,  c0:c1] * dt          # 각속도
            self.B_d[9:12, c0:c1] = self.B_c[9:12, c0:c1] * dt          # 선속도

            # 위치 파트 (2차 적분)
            self.B_d[0:3, c0:c1] = 0.5 * (self.wrapper.R_yaw.T @ self.B_c[6:9, c0:c1]) * (dt ** 2)
            self.B_d[3:6, c0:c1] = 0.5 * self.B_c[9:12, c0:c1] * (dt ** 2)

    def print_matrices(self):
        np.set_printoptions(precision=3, suppress=True, linewidth=150)
        nu = 3 * self.n_legs
        print("\n" + "="*60)
        print("💡 연속 시간 행렬 A_c (13x13)")
        print("="*60); print(self.A_c)

        print("\n" + "="*60)
        print(f"💡 연속 시간 입력 행렬 B_c (13x{nu})")
        print("="*60); print(self.B_c)

        print("\n" + "="*60)
        print("💡 이산 시간 행렬 A_d (13x13)")
        print("="*60); print(self.A_d)

        print("\n" + "="*60)
        print(f"💡 이산 시간 입력 행렬 B_d (13x{nu})")
        print("="*60); print(self.B_d)
        print("="*60 + "\n")
