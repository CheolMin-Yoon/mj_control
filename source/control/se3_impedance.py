r"""
$$
\begin{aligned}
A\dot{\tilde{V}}+D\tilde{V}+\mathrm{dexp}_{\lambda}^{-\top}K\lambda&=\tilde{F}
\qquad(\text{SE(3) 목표 임피던스}) \\
A_{\lambda}\ddot{\lambda}+D_{\lambda}\dot{\lambda}+K_{\lambda}\lambda&=\gamma
\qquad(\mathfrak{se}(3)\ \text{등가 오차 동역학})
\end{aligned}
\tag{55, 57}
$$
"""

import numpy as np

from source.dynamics.se3_pin import Ad, ad, dexp6, dexp6_dot, dexp6_inv, log6


class LieGroupImpedanceController:
    """Kim et al. TRO 2025 Lie group 임피던스, body frame {b} 기준 [linear; angular]

    인자:
        A: 목표 관성 (6, 6)
        D: 목표 감쇠 (6, 6)
        K: 목표 강성 (6, 6)
    """

    def __init__(self, A, D, K):
        self.A = np.asarray(A, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self._A_inv = np.linalg.inv(self.A)
        # 기록용
        self.lam = np.zeros(6)
        self.lam_dot = np.zeros(6)
        self.twist_err = np.zeros(6)

    def reference_acceleration(self, pose, twist, pose_des, twist_des, twist_des_dot, wrench, wrench_des):
        r"""
        $$
        \begin{aligned}
        \tilde{T}&=T^{-1}T_d,\qquad \tilde{V}=\mathrm{Ad}_{\tilde{T}}V_d-V,\qquad
        \lambda=\log\tilde{T},\qquad \dot{\lambda}=\mathrm{dexp}_{\lambda}^{-1}\tilde{V} \\
        \tilde{F}&=\mathrm{Ad}_{\tilde{T}}^{-\top}F_d-F,\qquad \gamma=\mathrm{dexp}_{\lambda}^{\top}\tilde{F} \\
        K_V&=\mathrm{dexp}_{\lambda}^{-1}\Big(A^{-1}D\,\mathrm{dexp}_{\lambda}+\tfrac{d}{dt}\mathrm{dexp}_{\lambda}\Big),\quad
        K_P=\mathrm{dexp}_{\lambda}^{-1}A^{-1}\mathrm{dexp}_{\lambda}^{-\top}K,\quad
        K_F=\mathrm{dexp}_{\lambda}^{-1}A^{-1}\mathrm{dexp}_{\lambda}^{-\top} \\
        \ddot{\lambda}_{\mathrm{ref}}&=-K_V\dot{\lambda}-K_P\lambda+K_F\gamma \\
        \dot{V}_{\mathrm{ref}}&=\mathrm{Ad}_{\tilde{T}}\dot{V}_d-\mathrm{dexp}_{\lambda}\ddot{\lambda}_{\mathrm{ref}}
        +\mathrm{ad}_{\tilde{V}}V-\tfrac{d}{dt}\mathrm{dexp}_{\lambda}\,\dot{\lambda}
        \end{aligned}
        \tag{44–64}
        $$

        pose·pose_des: world 기준 4×4, twist·wrench: 각 frame 자신 축 6D
        """
        rel_pose = np.linalg.solve(pose, pose_des)
        adjoint_rel = Ad(rel_pose)
        self.twist_err = adjoint_rel @ twist_des - twist
        self.lam = log6(rel_pose)

        dexp = dexp6(self.lam)
        dexp_inv = dexp6_inv(self.lam)
        self.lam_dot = dexp_inv @ self.twist_err
        dexp_dot = dexp6_dot(self.lam, self.lam_dot)

        wrench_err = np.linalg.solve(adjoint_rel.T, wrench_des) - wrench
        gamma = dexp.T @ wrench_err

        compliance = dexp_inv @ self._A_inv @ dexp_inv.T
        gain_v = dexp_inv @ (self._A_inv @ self.D @ dexp + dexp_dot)
        lam_ddot_ref = -gain_v @ self.lam_dot - compliance @ self.K @ self.lam + compliance @ gamma

        return adjoint_rel @ twist_des_dot - dexp @ lam_ddot_ref + ad(self.twist_err) @ twist - dexp_dot @ self.lam_dot

    def joint_torque(self, twist_ref_dot, jacobian_b, jdot_dq_b, mass_matrix, bias_forces, wrench):
        r"""
        $$
        \ddot{\theta}_{\mathrm{ref}}=J_b^{-1}\big(\dot{V}_{\mathrm{ref}}-\dot{J}_b\dot{\theta}\big),\qquad
        \tau=M\ddot{\theta}_{\mathrm{ref}}+C\dot{\theta}+g-J_b^{\top}F
        \tag{65, 66}
        $$
        """
        joint_acc_ref = np.linalg.solve(jacobian_b, twist_ref_dot - jdot_dq_b)
        return mass_matrix @ joint_acc_ref + bias_forces - jacobian_b.T @ wrench
