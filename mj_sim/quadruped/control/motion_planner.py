import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Pinocchio_Wrapper


HEIGHT_SWING = 0.08


class MotionPlanner:
    """Swing foot 궤적 (Bezier) + Raibert footstep heuristic.

    LegController가 swing 시점에 호출 — phase로 발끝 (pos, vel, acc),
    swing 진입 시 다음 착지점(touchdown)을 받는다.
    """

    def __init__(self, wrapper: 'Pinocchio_Wrapper',
                 raibert_kp=(0.05, 0.05),
                 raibert_kv_scale=(1.1, 1.0)):
        """
        raibert_kp       : (kp_x, kp_y) — CoM 위치 오차 게인 (현재는 velocity-only 보정이라 reserve)
        raibert_kv_scale : (kv_x, kv_y) — time_constant에 곱하는 속도 게인
        """
        self.wrapper          = wrapper
        self.raibert_kp       = np.asarray(raibert_kp)
        self.raibert_kv_scale = np.asarray(raibert_kv_scale)


    # 4차 Bezier XY + 5차 Bezier Z — boundary v=0, peak Z 튜닝
    def compute_bezier_trajectory(self, phase, pos_start, pos_end, duration, step_height):
        """
        phase       : 0 ~ 1
        pos_start   : (3,) swing 시작
        pos_end     : (3,) swing 끝 (touchdown)
        duration    : 전체 swing 시간 [s]
        step_height : 발 peak 높이 [m]
        Returns: pos, vel, acc — 각 (3,)
        """
        t   = phase
        omt = 1 - t

        # XY: 4차 Bezier — 제어점 [s, s, mid, e, e] → 시작/끝 속도 0
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

        # Z: 5차 Bezier — 제어점 비율로 가파름 튜닝 (1.2, 0.5)
        h  = step_height
        Pz = np.array([pos_start[2], pos_start[2], h*1.2, h*0.5, pos_end[2], pos_end[2]])

        z       = (omt**5 * Pz[0] + 5*t*omt**4 * Pz[1] + 10*t**2*omt**3 * Pz[2]
                   + 10*t**3*omt**2 * Pz[3] + 5*t**4*omt * Pz[4] + t**5 * Pz[5])
        dz_dp   = 5 * (omt**4 * (Pz[1]-Pz[0]) + 4*t*omt**3 * (Pz[2]-Pz[1])
                       + 6*t**2*omt**2 * (Pz[3]-Pz[2]) + 4*t**3*omt * (Pz[4]-Pz[3])
                       + t**4 * (Pz[5]-Pz[4]))
        dz2_dp2 = 20 * (omt**3 * (Pz[2]-2*Pz[1]+Pz[0]) + 3*t*omt**2 * (Pz[3]-2*Pz[2]+Pz[1])
                        + 3*t**2*omt * (Pz[4]-2*Pz[3]+Pz[2]) + t**3 * (Pz[5]-2*Pz[4]+Pz[3]))

        pos = np.array([xy[0],       xy[1],       z])
        vel = np.array([dxy_dp[0],   dxy_dp[1],   dz_dp])    / duration
        acc = np.array([dxy2_dp2[0], dxy2_dp2[1], dz2_dp2])  / duration**2
        return pos, vel, acc


    # Raibert heuristic — 다음 swing foot touchdown (world)
    def compute_raibert_heuristic(self, leg: str,
                                  vx_des: float, vy_des: float, yaw_rate_des: float):
        wrapper = self.wrapper
        base_pos_W = wrapper.p_base_W
        com_pos_W  = wrapper.p_com_W
        com_vel_W  = wrapper.v_com_W
        R_WH       = wrapper.R_WH

        # hip world 위치 (base + R_WH @ hip_offset)
        hip_offset = wrapper.get_hip_offset(leg)
        hip_pos_W  = base_pos_W + R_WH @ hip_offset

        # LIPM/CP time constant
        h_com         = com_pos_W[2]
        omega         = np.sqrt(9.81 / h_com)
        time_constant = 1.0 / omega

        kv_x, kv_y = self.raibert_kv_scale * time_constant

        # nominal: hip 아래, z는 ground 직전
        pos_nominal = np.array([hip_pos_W[0], hip_pos_W[1], 0.01])

        # CoM 속도 오차 보정
        vel_correction = np.array([
            kv_x * (com_vel_W[0] - vx_des),
            kv_y * (com_vel_W[1] - vy_des),
            0.0,
        ])

        # yaw 회전 보정 — 다음 step에서 base가 dθ 추가 회전 가정
        dtheta   = yaw_rate_des * time_constant
        c, s     = np.cos(dtheta), np.sin(dtheta)
        r_x, r_y = pos_nominal[:2] - base_pos_W[:2]
        rot_correction = np.array([
            (c - 1.0) * r_x - s         * r_y,
            s         * r_x + (c - 1.0) * r_y,
            0.0,
        ])

        return pos_nominal + vel_correction + rot_correction
