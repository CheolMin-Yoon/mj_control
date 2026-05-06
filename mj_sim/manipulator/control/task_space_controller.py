import numpy as np
import pinocchio as pin

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Pinocchio_Wrapper


class TaskSpaceController:
    """Free-end task-space tracking — swing foot, hand 등 비접촉 EE 전용.

    floating-base humanoid의 contact-supporting DOF에는 사용 금지.
    그쪽은 WholeBodyController가 contact constraint와 함께 처리.

    NOTE: compute_error 반환값(pose_err, twist_err)은 instance 버퍼 view.
    동일 step 내 1회 사용 가정. 보존 필요 시 호출자가 .copy() 해야 함.
    """

    def __init__(self, wrapper: 'Pinocchio_Wrapper'):
        self.wrapper = wrapper

        # 게인 
        wn_pos = 20.0
        wn_ori = 20.0
        self.Kp_pos = wn_pos**2 * np.eye(3)
        self.Kp_ori = wn_ori**2 * np.eye(3)
        self.Kd_lin = 2 * wn_pos * np.eye(3)
        self.Kd_ang = 2 * wn_ori * np.eye(3)

        # DLS 파라미터 (Λ_inv + λ²·I 형태로 특이자세 안전)
        self.I        = np.eye(6)
        self.damping2 = 1e-3 ** 2     # λ² (제곱 형태로 저장)

        # 버퍼 (매 step 재사용)
        self.pose_err  = np.zeros(6)
        self.twist_err = np.zeros(6)
        self.F_des     = np.zeros(6)


    # 오차 계산 (current vs desired → 6D pose / twist 오차)
    def compute_error(self, pos_curr, R_curr, twist_curr, pos_des, R_des, twist_des):
        self.pose_err[:3] = pos_des - pos_curr
        self.pose_err[3:] = pin.log3(R_des @ R_curr.T)
        self.twist_err[:] = twist_des - twist_curr
        return self.pose_err, self.twist_err


    # OSC impedance 토크 (free-end task only — class docstring 참조)
    def compute_impedance_torque(self, pose_err, twist_err, J, M_inv, nle):
        # Λ⁻¹ = J·M⁻¹·Jᵀ + λ²·I  → 항상 invertible (특이자세 안전)
        Lambda_inv = J @ M_inv @ J.T + self.damping2 * self.I
        Lambda     = np.linalg.inv(Lambda_inv)

        self.F_des[:3] = self.Kp_pos @ pose_err[:3] + self.Kd_lin @ twist_err[:3]
        self.F_des[3:] = self.Kp_ori @ pose_err[3:] + self.Kd_ang @ twist_err[3:]

        return J.T @ (Lambda @ self.F_des) + nle


    # Damped Least Squares IK (velocity-level): V_des = twist_des + Kp·pose_err
    def compute_DLS(self, pose_err, twist_des, J):
        V_pos_des = twist_des[:3] + self.Kp_pos @ pose_err[:3]
        V_ori_des = twist_des[3:] + self.Kp_ori @ pose_err[3:]
        V_des     = np.concatenate([V_pos_des, V_ori_des])

        JJt    = J @ J.T + self.damping2 * self.I
        dq_des = J.T @ np.linalg.solve(JJt, V_des)
        return dq_des
