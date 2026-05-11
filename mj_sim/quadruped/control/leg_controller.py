import numpy as np
from dataclasses import dataclass

from core import Pinocchio_Wrapper, Mujoco_Kernel, LEG_KEYS, LEG_IDX
from .gait_scheduler import GaitScheduler
from .motion_planner import MotionPlanner, HEIGHT_SWING


@dataclass
class LegOutput:
    tau:     np.ndarray   # (3,) joint torque
    pos_des: np.ndarray
    pos_now: np.ndarray
    vel_des: np.ndarray
    vel_now: np.ndarray


class LegController:
    """reference (go2-convex-mpc/leg_controller.py) 정설:
       - stance: tau = J.T @ (-f_grf)                                                (GRF 단독)
       - swing : tau = J.T @ (KP·e_p + KD·e_v + Lambda·(a_des - Jdot·dq)) + (C·dq + g)[leg]
    """
    KP_SWING = np.diag([400.0, 400.0, 400.0])
    KD_SWING = np.diag([ 75.0,  75.0,  75.0])

    def __init__(self, wrapper: Pinocchio_Wrapper, gait: GaitScheduler,
                 motion_planner: MotionPlanner, kernel: Mujoco_Kernel,
                 vx_des=0.0, vy_des=0.0, yaw_rate_des=0.0):
        self.wrapper        = wrapper
        self.gait           = gait
        self.motion_planner = motion_planner

        self.vx_des       = vx_des
        self.vy_des       = vy_des
        self.yaw_rate_des = yaw_rate_des

        self._last_mask = np.ones(4, dtype=np.int32)
        self._p_takeoff = np.zeros((3, 4))   # swing 진입 시점 발 위치 캐시
        self._p_target  = np.zeros((3, 4))   # Raibert touchdown target

        del kernel   # 호환 위해 받지만 사용 안 함

    def compute_tau(self, t: float, f_grf=None, pos_des_W=None) -> np.ndarray:
        """t: 현재 시각, f_grf: (12,) MPC GRF world frame [fl, fr, rl, rr],
           pos_des_W: (3,) Raibert pos FB용 desired CoM 위치 (없으면 무시)."""
        wrapper       = self.wrapper
        planner       = self.motion_planner
        mask, swing_p = self.gait.compute_gait_state(t)
        tau_all = np.zeros(12)

        M_inv = wrapper.M_inv
        nle   = wrapper.nle

        for leg in LEG_KEYS:
            idx = LEG_IDX[leg]
            col = idx * 3

            J         = wrapper.compute_J_W(leg)        # (6, 18)
            J3        = J[:3, 6:][:, col : col + 3]     # (3, 3) — 해당 leg 3 joints
            J3_full   = J[:3, :]                        # (3, 18)

            oMf, v_lin, _ = wrapper.compute_ee_state_W(leg)
            pos_now = oMf.translation
            vel_now = v_lin

            if mask[idx] == 0:   # ── swing ──
                if self._last_mask[idx] == 1:
                    self._p_takeoff[:, idx] = pos_now
                    self._p_target[:, idx]  = planner.compute_raibert_heuristic(
                        leg, self.vx_des, self.vy_des, self.yaw_rate_des,
                        self.gait.stance_time, self.gait.swing_time,
                        pos_des_W=pos_des_W,
                    )

                pos_des, vel_des, acc_des = planner.compute_bezier_trajectory(
                    swing_p[idx],
                    self._p_takeoff[:, idx],
                    self._p_target[:, idx],
                    self.gait.swing_time,
                    HEIGHT_SWING,
                )

                # operational space inertia + Jdot·dq 보상
                Lambda  = np.linalg.inv(J3_full @ M_inv @ J3_full.T)
                Jdot_dq = wrapper.compute_Jdot_dq_W(leg)[:3]
                f_ff    = Lambda @ (acc_des - Jdot_dq)

                f_task  = (self.KP_SWING @ (pos_des - pos_now)
                           + self.KD_SWING @ (vel_des - vel_now)
                           + f_ff)
                tau_leg = J3.T @ f_task + nle[6 + col : 6 + col + 3]

            else:   # ── stance ──
                f = f_grf[col : col + 3] if f_grf is not None else np.zeros(3)
                tau_leg = J3.T @ (-f)

            tau_all[col : col + 3] = tau_leg
            self._last_mask[idx] = mask[idx]

        return tau_all
