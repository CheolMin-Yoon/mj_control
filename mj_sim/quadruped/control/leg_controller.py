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
    """4발 토크 합성:
       tau = tau_ff + tau_fb
       - tau_ff: MPC GRF → joint torque (J.T @ f_grf_W)
       - tau_fb: stance면 joint-space PD(q_nom), swing이면 task-space PD(Bezier traj)
       f_grf=None이면 tau_ff=0, fb만 (MPC 미사용 fallback).
    """
    KP_STANCE = np.diag([80.0, 80.0, 80.0])
    KD_STANCE = np.diag([10.0, 10.0, 10.0])

    KP_SWING  = np.diag([100.0, 100.0, 100.0])
    KD_SWING  = np.diag([10.0,  10.0,  10.0])

    def __init__(self, wrapper: Pinocchio_Wrapper, gait: GaitScheduler,
                 motion_planner: MotionPlanner, kernel: Mujoco_Kernel,
                 vx_des=0.0, vy_des=0.0, yaw_rate_des=0.0):
        self.wrapper        = wrapper
        self.gait           = gait
        self.motion_planner = motion_planner

        # 외부에서 직접 갱신 (속도 명령 — body frame)
        self.vx_des       = vx_des
        self.vy_des       = vy_des
        self.yaw_rate_des = yaw_rate_des

        self._last_mask = np.ones(4, dtype=np.int32)
        self._p_takeoff = np.zeros((3, 4))   # swing 진입 시점 발 위치 캐시
        self._p_target  = np.zeros((3, 4))   # Raibert touchdown target

        # stance nominal: MJCF keyframe[0] joint 자세
        if kernel.q_keyframe is not None:
            self._q_nom = kernel.q_keyframe[7:].copy()
        else:
            self._q_nom = np.zeros(12)

    def compute_tau(self, t: float, f_grf=None) -> np.ndarray:
        """t: 현재 시각, f_grf: (12,) MPC GRF world frame [fl, fr, rl, rr], None이면 fb만."""
        wrapper       = self.wrapper
        planner       = self.motion_planner
        mask, swing_p = self.gait.compute_gait_state(t)
        tau_all = np.zeros(12)

        for leg in LEG_KEYS:
            idx = LEG_IDX[leg]
            col = idx * 3

            J = wrapper.compute_J_W(leg)
            J3 = J[:3, 6:][:, col : col + 3]

            oMf, v_lin, _ = wrapper.compute_ee_state_W(leg)
            pos_now = oMf.translation
            vel_now = v_lin

            # feedforward: MPC GRF → joint torque
            tau_ff = J3.T @ f_grf[col : col + 3] if f_grf is not None else 0.0

            # feedback: stance면 joint PD, swing이면 task-space PD
            if mask[idx] == 0:   # ── swing ──
                if self._last_mask[idx] == 1:
                    self._p_takeoff[:, idx] = pos_now
                    self._p_target[:, idx]  = planner.compute_raibert_heuristic(
                        leg, self.vx_des, self.vy_des, self.yaw_rate_des
                    )

                pos_des, vel_des, _ = planner.compute_bezier_trajectory(
                    swing_p[idx],
                    self._p_takeoff[:, idx],
                    self._p_target[:, idx],
                    self.gait.swing_time,
                    HEIGHT_SWING,
                )
                f_task = self.KP_SWING @ (pos_des - pos_now) + self.KD_SWING @ (vel_des - vel_now)
                tau_fb = J3.T @ f_task

            else:   # ── stance ──
                q_now  = wrapper.q[7 + col : 7 + col + 3]
                dq_now = wrapper.dq[6 + col : 6 + col + 3]
                q_des  = self._q_nom[col : col + 3]
                tau_fb = self.KP_STANCE @ (q_des - q_now) - self.KD_STANCE @ dq_now

            tau_all[col : col + 3] = tau_ff + tau_fb
            self._last_mask[idx] = mask[idx]

        return tau_all
