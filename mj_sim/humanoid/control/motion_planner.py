import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Pinocchio_Wrapper
    from control import GaitScheduler, TrajOptimizer


class MotionPlanner:

    def __init__(self, wrapper: 'Pinocchio_Wrapper',
                 traj_optimizer: 'TrajOptimizer',
                 gait_scheduler: 'GaitScheduler',
                 raibert_kp=(0.05, 0.05),
                 raibert_kv_scale=(1.1, 1.0)):
        """
        raibert_kp       : (kp_x, kp_y) CoM 위치 오차 게인
        raibert_kv_scale : (kv_x, kv_y) CoM 속도 오차 게인 — time_constant에 곱해서 사용
                           기본값은 MIT Raibert/ALIP 관습 (sagittal 방향 1.1배 가중)
        """
        self.wrapper        = wrapper
        self.traj_optimizer = traj_optimizer
        self.gait_scheduler = gait_scheduler
        self.raibert_kp       = np.asarray(raibert_kp)
        self.raibert_kv_scale = np.asarray(raibert_kv_scale)


    # Swing foot 궤적 (5차 Bezier Z + 4차 Bezier XY)
    def compute_bezier_trajectory(self, phase, pos_start, pos_end, duration, step_height):
        """
        phase       : 0 ~ 1
        pos_start   : (3,) [x, y, z] swing 시작
        pos_end     : (3,) [x, y, z] swing 끝
        duration    : 전체 swing 시간 [s]
        step_height : 발 들어올리는 peak 높이 [m]
        Returns: pos / vel / acc — 각 (3,)
        """
        t   = phase
        omt = 1 - t

        # XY: 4차 Bezier, 제어점 [s, s, mid, e, e] → 시작/끝 속도 0
        xy_s = pos_start[:2]
        xy_e = pos_end[:2]
        Pxy  = np.stack([xy_s, xy_s, (xy_s + xy_e) / 2, xy_e, xy_e])

        xy       = (omt**4 * Pxy[0] + 4*t*omt**3 * Pxy[1] + 6*t**2*omt**2 * Pxy[2]
                    + 4*t**3*omt * Pxy[3] + t**4 * Pxy[4])
        dxy_dp   = 4 * (omt**3 * (Pxy[1]-Pxy[0]) + 3*t*omt**2 * (Pxy[2]-Pxy[1])
                        + 3*t**2*omt * (Pxy[3]-Pxy[2]) + t**3 * (Pxy[4]-Pxy[3]))
        dxy2_dp2 = 12 * (omt**2 * (Pxy[2]-2*Pxy[1]+Pxy[0])
                         + 2*t*omt * (Pxy[3]-2*Pxy[2]+Pxy[1])
                         + t**2 * (Pxy[4]-2*Pxy[3]+Pxy[2]))

        # Z: 5차 Bezier, 제어점 비율로 궤적 가파름 튜닝 (1.2, 0.5)
        h  = step_height
        Pz = np.array([pos_start[2], pos_start[2], h*1.2, h*0.5, pos_end[2], pos_end[2]])

        z       = (omt**5 * Pz[0] + 5*t*omt**4 * Pz[1] + 10*t**2*omt**3 * Pz[2]
                   + 10*t**3*omt**2 * Pz[3] + 5*t**4*omt * Pz[4] + t**5 * Pz[5])
        dz_dp   = 5 * (omt**4 * (Pz[1]-Pz[0]) + 4*t*omt**3 * (Pz[2]-Pz[1])
                       + 6*t**2*omt**2 * (Pz[3]-Pz[2]) + 4*t**3*omt * (Pz[4]-Pz[3])
                       + t**4 * (Pz[5]-Pz[4]))
        dz2_dp2 = 20 * (omt**3 * (Pz[2]-2*Pz[1]+Pz[0]) + 3*t*omt**2 * (Pz[3]-2*Pz[2]+Pz[1])
                        + 3*t**2*omt * (Pz[4]-2*Pz[3]+Pz[2]) + t**3 * (Pz[5]-2*Pz[4]+Pz[3]))

        # phase 미분 → 물리 시간 미분
        pos = np.array([xy[0],            xy[1],            z])
        vel = np.array([dxy_dp[0],        dxy_dp[1],        dz_dp])    / duration
        acc = np.array([dxy2_dp2[0],      dxy2_dp2[1],      dz2_dp2])  / duration**2

        return pos, vel, acc


    # Raibert heuristic — 다음 swing foot touchdown 위치 (안 넘어지기 위해)
    def compute_raibert_heuristic(self, leg: str):
        # 현재 상태
        base_pos_curr = self.wrapper.base_pos
        com_pos_curr  = self.wrapper.com_pos_world
        com_vel_curr  = self.wrapper.com_vel_world
        R_z           = self.wrapper.R_z

        # 목표 (traj_optimizer 시그니처: 5-tuple 반환 가정 — 추후 정의)
        pos_x_des, pos_y_des, vel_x_des, vel_y_des, yaw_rate_des = \
            self.traj_optimizer.compute_desired_command()

        # hip world 위치 (base + R_z @ hip_offset_in_base)
        hip_offset    = self.wrapper.get_hip_offset(leg)
        hip_pos_world = base_pos_curr + R_z @ hip_offset

        # Capture point time constant
        h_com         = com_pos_curr[2]
        omega         = np.sqrt(9.81 / h_com)
        time_constant = 1.0 / omega

        # 게인 (constructor에서 주입)
        kp_x, kp_y = self.raibert_kp
        kv_x, kv_y = self.raibert_kv_scale * time_constant

        # nominal: 현재 hip의 (x, y), z는 ground 직전
        pos_nominal = np.array([hip_pos_world[0], hip_pos_world[1], 0.01])

        # CoM 위치/속도 보정
        pos_correction = np.array([
            kp_x * (com_pos_curr[0] - pos_x_des),
            kp_y * (com_pos_curr[1] - pos_y_des),
            0.0,
        ])
        vel_correction = np.array([
            kv_x * (com_vel_curr[0] - vel_x_des),
            kv_y * (com_vel_curr[1] - vel_y_des),
            0.0,
        ])

        # yaw 회전 보정 — exact R_z(dθ)·r 적용 (small-angle 가정 없음)
        # 다음 step에서 base가 dθ 추가 회전 시 hip의 새 위치: base + R_z(dθ)·(hip_world - base)
        dtheta = yaw_rate_des * time_constant
        c, s   = np.cos(dtheta), np.sin(dtheta)
        r_x, r_y = pos_nominal[:2] - base_pos_curr[:2]
        rot_correction = np.array([
            (c - 1.0) * r_x - s         * r_y,
            s         * r_x + (c - 1.0) * r_y,
            0.0,
        ])

        return pos_nominal + pos_correction + vel_correction + rot_correction
