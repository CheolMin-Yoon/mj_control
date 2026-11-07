r"""
$$
\begin{aligned}
\tilde{T}&=T^{-1}T_d,\qquad \lambda=\log(\tilde{T}),\qquad V=J_b\dot{q} \\
\tilde{V}&=\mathrm{Ad}_{\tilde{T}}V_d-V,\qquad \dot{\lambda}=\mathrm{dexp}_{\lambda}^{-1}\tilde{V}
\end{aligned}
\tag{44, 45, 47, 48}
$$
"""

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from source.dynamics.se3_casadi import dexp6, dexp6_dot, dexp6_inv

TERM_NAMES = ("lam", "lam_dot", "dexp", "dexp_dot", "mass_matrix", "bias_forces", "jacobian_b", "jdot_dq_b")


def se3_impedance_terms(model, frame_name, name="se3_impedance_terms"):
    r"""
    $$
    \begin{aligned}
    (q,\dot{q},T_d,V_d)\ \mapsto\ &\big(\lambda,\ \dot{\lambda},\ \mathrm{dexp}_{\lambda},\
    \tfrac{d}{dt}\mathrm{dexp}_{\lambda},\ M,\ C\dot{q}+g,\ J_b,\ \dot{J}_b\dot{q}\big) \\
    T_d&=\begin{bmatrix}R_d&p_d\\0&1\end{bmatrix},\qquad
    \texttt{pose\_des}=\big[\mathrm{vec}(R_d);\,p_d\big]\in\mathbb{R}^{12}\ (\text{column-major})
    \end{aligned}
    $$
    """
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    frame_id = model.getFrameId(frame_name)

    q = ca.SX.sym("q", model.nq)
    dq = ca.SX.sym("dq", model.nv)
    pose_des = ca.SX.sym("pose_des", 12)
    twist_des = ca.SX.sym("twist_des", 6)

    cpin.forwardKinematics(cmodel, cdata, q, dq)
    cpin.computeJointJacobians(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    jacobian_b = cpin.getFrameJacobian(cmodel, cdata, frame_id, pin.ReferenceFrame.LOCAL)
    cpin.computeJointJacobiansTimeVariation(cmodel, cdata, q, dq)
    jdot_dq_b = cpin.getFrameJacobianTimeVariation(cmodel, cdata, frame_id, pin.ReferenceFrame.LOCAL) @ dq
    # crba는 상삼각만 채움 → 대칭화
    mass_upper = ca.triu(cpin.crba(cmodel, cdata, q))
    mass_matrix = mass_upper + ca.triu(mass_upper, False).T
    bias_forces = cpin.nonLinearEffects(cmodel, cdata, q, dq)

    pose = cdata.oMf[frame_id]
    rot, pos = pose.rotation, pose.translation
    rot_des, pos_des = ca.reshape(pose_des[:9], 3, 3), pose_des[9:]
    rel_rot = rot.T @ rot_des
    rel_pos = rot.T @ (pos_des - pos)
    lam = cpin.log6(cpin.SE3(rel_rot, rel_pos)).vector

    twist = jacobian_b @ dq
    adjoint_rel = ca.blockcat([[rel_rot, ca.skew(rel_pos) @ rel_rot], [ca.SX.zeros(3, 3), rel_rot]])
    lam_dot = dexp6_inv(lam) @ (adjoint_rel @ twist_des - twist)

    outputs = (lam, lam_dot, dexp6(lam), dexp6_dot(lam, lam_dot), mass_matrix, bias_forces, jacobian_b, jdot_dq_b)
    return ca.Function(
        name, [q, dq, pose_des, twist_des], list(outputs), ["q", "dq", "pose_des", "twist_des"], list(TERM_NAMES)
    )
