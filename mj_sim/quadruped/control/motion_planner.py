import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Pinocchio_Wrapper


HEIGHT_SWING = 0.1


class MotionPlanner:
    """Swing foot 궤적 (Bezier) + Raibert footstep heuristic.

    LegController가 swing 시점에 호출 — phase로 발끝 (pos, vel, acc),
    swing 진입 시 다음 착지점(touchdown)을 받는다.
    """

    def __init__(self, wrapper: 'Pinocchio_Wrapper'):
        self.wrapper = wrapper


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
                                  vx_des: float, vy_des: float, yaw_rate_des: float,
                                  t_stance: float, t_swing: float,
                                  pos_des_W=None):
        wrapper = self.wrapper
        base_pos_W = wrapper.p_base_W
        com_pos_W  = wrapper.p_com_W
        com_vel_W  = wrapper.v_com_W
        R_WH       = wrapper.R_WH

        # hip world 위치 (base + R_WH @ hip_offset)
        hip_offset = wrapper.get_hip_offset(leg)
        hip_pos_W  = base_pos_W + R_WH @ hip_offset

        # gait 기반 prediction horizon (Cheetah/go2-convex-mpc 컨벤션)
        T          = t_swing + 0.5 * t_stance
        pred_time  = 0.5 * T
        kv_x, kv_y = 0.4 * T, 0.2 * T
        kp_x, kp_y = 0.1, 0.05

        # body → world 속도 명령
        v_des_W = R_WH @ np.array([vx_des, vy_des, 0.0])

        # nominal: hip 아래, z는 ground 직전
        pos_nominal = np.array([hip_pos_W[0], hip_pos_W[1], 0.01])

        # feedforward: mid-flight 까지 hip 진행 거리
        ff = v_des_W * pred_time
        ff[2] = 0.0

        # CoM 속도 오차 보정 (capture point)
        vel_correction = np.array([
            kv_x * (com_vel_W[0] - v_des_W[0]),
            kv_y * (com_vel_W[1] - v_des_W[1]),
            0.0,
        ])

        # CoM 위치 오차 보정 (reference의 pos_correction_term)
        if pos_des_W is not None:
            pos_correction = np.array([
                kp_x * (com_pos_W[0] - pos_des_W[0]),
                kp_y * (com_pos_W[1] - pos_des_W[1]),
                0.0,
            ])
        else:
            pos_correction = np.zeros(3)

        # yaw 회전 보정 — pred_time 동안 base가 dθ 추가 회전 가정
        dtheta   = yaw_rate_des * pred_time
        c, s     = np.cos(dtheta), np.sin(dtheta)
        r_x, r_y = pos_nominal[:2] - base_pos_W[:2]
        rot_correction = np.array([
            (c - 1.0) * r_x - s         * r_y,
            s         * r_x + (c - 1.0) * r_y,
            0.0,
        ])

        return pos_nominal + ff + vel_correction + pos_correction + rot_correction
