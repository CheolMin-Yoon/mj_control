import numpy as np
from dataclasses import dataclass

from core import Pinocchio_Wrapper, Mujoco_Kernel, LEG_KEYS, LEG_IDX
from .gait_scheduler import GaitScheduler, HEIGHT_SWING


@dataclass
class LegOutput:
    tau:     np.ndarray   # (3,) joint torque
    pos_des: np.ndarray
    pos_now: np.ndarray
    vel_des: np.ndarray
    vel_now: np.ndarray


class LegController:
    """4발 PD 제어:
       - stance: joint-space PD (q_nom 기준)
       - swing : task-space PD (발끝 world 위치)
    """
    KP_STANCE = np.diag([80.0, 80.0, 80.0])
    KD_STANCE = np.diag([10.0, 10.0, 10.0])

    KP_SWING  = np.diag([100.0, 100.0, 100.0])
    KD_SWING  = np.diag([10.0,  10.0,  10.0])

    def __init__(self, wrapper: Pinocchio_Wrapper, gait: GaitScheduler, kernel: Mujoco_Kernel):
        self.wrapper    = wrapper
        self.gait       = gait
        self._last_mask = np.ones(4, dtype=np.int32)   # 초기: 전부 stance
        self._traj      = {}    # leg -> eval_at 클로저
        self._t_takeoff = {}    # leg -> 이륙 시각

        # stance nominal: MJCF keyframe[0] joint 자세 (pinocchio neutral과 다름)
        if kernel.q_keyframe is not None:
            self._q_nom = kernel.q_keyframe[7:].copy()   # (12,)
        else:
            self._q_nom = np.zeros(12)

    def compute(self, t: float) -> np.ndarray:
        wrapper = self.wrapper
        gait    = self.gait
        mask    = gait.get_mask(t)
        tau_all = np.zeros(12)

        for leg in LEG_KEYS:
            idx = LEG_IDX[leg]
            col = idx * 3   # 0,3,6,9 for FL,FR,RL,RR

            J = wrapper.compute_J_W(leg)
            J3 = J[:3, 6:][:, col : col + 3]   # (3, 3) 해당 다리 3DOF

            oMf, v_lin, _ = wrapper.compute_ee_state_W(leg)
            pos_now = oMf.translation
            vel_now = v_lin

            if mask[idx] == 0:   # ── swing ──
                if self._last_mask[idx] == 1:
                    p0 = pos_now.copy()
                    pf = p0.copy()
                    self._traj[leg]      = gait.make_swing_traj(p0, pf, HEIGHT_SWING)
                    self._t_takeoff[leg] = t

                dt     = t - self._t_takeoff.get(leg, t)
                eval_at = self._traj.get(leg)
                if eval_at is None:
                    continue

                pos_des, vel_des, _ = eval_at(dt)
                f_task = self.KP_SWING @ (pos_des - pos_now) + self.KD_SWING @ (vel_des - vel_now)
                tau3   = J3.T @ f_task

            else:   # ── stance ──
                q_now  = wrapper.q[7 + col : 7 + col + 3]
                dq_now = wrapper.dq[6 + col : 6 + col + 3]
                q_des  = self._q_nom[col : col + 3]
                tau3   = self.KP_STANCE @ (q_des - q_now) - self.KD_STANCE @ dq_now

            tau_all[col : col + 3] = tau3
            self._last_mask[idx] = mask[idx]

        return tau_all
