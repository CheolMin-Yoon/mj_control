"""한 frame 작업공간 추종: OSC 임피던스 torque, DLS 속도 IK, DLS QP 비용"""

import numpy as np
import pinocchio as pin


class TaskSpaceController:
    """비접촉 frame 추종, 오차는 [위치; 회전 log] world 축 6D

    가속도 PD 이득 kp = wn², kd = 2·wn(임계 감쇠)이고 물리 강성·감쇠가 아님

    인자:
        nv: 관절 수
        wn_pos: 위치 오차 고유진동수 [rad/s]
        wn_ori: 자세 오차 고유진동수 [rad/s]
        task_regularization: Λ⁻¹·JJᵀ·JᵀJ 대각 정규화 λ², 특이자세 대비
    """

    def __init__(self, nv, wn_pos=10.0, wn_ori=10.0, task_regularization=1e-6):
        self.kp_lin = wn_pos**2 * np.eye(3)
        self.kp_ang = wn_ori**2 * np.eye(3)
        self.kd_lin = 2 * wn_pos * np.eye(3)
        self.kd_ang = 2 * wn_ori * np.eye(3)
        self.task_regularization = task_regularization

        self._task_eye = np.eye(6)
        self._joint_eye = np.eye(nv)
        # 버퍼, 같은 step 안 1회 사용 가정 → 보존은 호출자 copy
        self.pose_err = np.zeros(6)
        self.twist_err = np.zeros(6)
        self.task_acc_des = np.zeros(6)

    def tracking_error(self, pos_w, mat_w, twist_w, pos_des_w, mat_des_w, twist_des_w):
        """(pose_err (6,), twist_err (6,)) 반환, 회전 오차 log3(R_des Rᵀ)는 world 축"""
        self.pose_err[:3] = pos_des_w - pos_w
        self.pose_err[3:] = pin.log3(mat_des_w @ mat_w.T)
        self.twist_err[:] = twist_des_w - twist_w
        return self.pose_err, self.twist_err

    def impedance_torque(self, pose_err, twist_err, jacobian_w, mass_matrix_inv, bias_forces):
        """τ = Jᵀ Λ a_des + bias, a_des = kp·pose_err + kd·twist_err, J̇q̇ 보상 없음"""
        Lambda = np.linalg.inv(jacobian_w @ mass_matrix_inv @ jacobian_w.T + self.task_regularization * self._task_eye)
        self.task_acc_des[:3] = self.kp_lin @ pose_err[:3] + self.kd_lin @ twist_err[:3]
        self.task_acc_des[3:] = self.kp_ang @ pose_err[3:] + self.kd_ang @ twist_err[3:]
        return jacobian_w.T @ (Lambda @ self.task_acc_des) + bias_forces

    def dls_joint_vel(self, pose_err, twist_des, jacobian_w):
        """DLS 관절 속도 dq = Jᵀ(JJᵀ + λ²I)⁻¹ V, V = twist_des + kp·pose_err"""
        V_des = self._feedback_twist(pose_err, twist_des)
        JJt = jacobian_w @ jacobian_w.T + self.task_regularization * self._task_eye
        return jacobian_w.T @ np.linalg.solve(JJt, V_des)

    def dls_qp_cost(self, pose_err, twist_des, jacobian_w):
        """min ½dqᵀH dq + cᵀdq의 (H, c) 반환, ‖J dq − V‖² + λ²‖dq‖²와 같은 해"""
        V_des = self._feedback_twist(pose_err, twist_des)
        H = jacobian_w.T @ jacobian_w + self.task_regularization * self._joint_eye
        return H, -jacobian_w.T @ V_des

    def _feedback_twist(self, pose_err, twist_des):
        return np.concatenate(
            [
                twist_des[:3] + self.kp_lin @ pose_err[:3],
                twist_des[3:] + self.kp_ang @ pose_err[3:],
            ]
        )
