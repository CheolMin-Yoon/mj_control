"""Indy7 v3 Lie group 임피던스(Kim et al. TRO 2025) 데모

--experiment push       home 자세 유지, viewer에서 Ctrl + 마우스 drag로 밀면 복원 (F/T 없음 → 관성은 로봇 고유)
--experiment rendering  10 cm 계단 목표, D = K = 20, --inertia로 A 선택, 이상 2차계 응답과 비교 (Sec. IV-A)
--experiment half_turn  TCP z축 180° 목표, A = 4, D = 150, K = 40 (Sec. IV-B)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.control.se3_impedance import LieGroupImpedanceController  # noqa: E402
from source.dynamics import PinocchioWrapper  # noqa: E402
from source.robots.indy7_v3.constants import (  # noqa: E402
    INDY7_V3_FRAME_NAMES,
    INDY7_V3_JOINT_NAMES,
    INDY7_V3_SCENE_XML,
    INDY7_V3_URDF,
)
from source.sim import DataLogger, MujocoKernel, SimScheduler  # noqa: E402

CTRL_HZ = 1000


def impedance(translation, rotation):
    """(A, D, K) 6×6, 각 인자는 병진·회전 (a, d, k) 스칼라"""
    return tuple(np.diag([translation[i]] * 3 + [rotation[i]] * 3) for i in range(3))


def target_pose(experiment, home_pose):
    """실험별 world 기준 목표 4×4"""
    target = home_pose.copy()
    if experiment == "rendering":
        target[:3, 3] += np.array([0.1, 0.0, 0.0])
    elif experiment == "half_turn":
        target[:3, :3] = home_pose[:3, :3] @ pin.exp3(np.array([0.0, 0.0, np.pi]))
    return target


def ideal_response(a, d, k, e0, times):
    """a ë + d ė + k e = 0, e(0) = e0, ė(0) = 0"""
    sol = solve_ivp(lambda t, x: [x[1], -(d * x[1] + k * x[0]) / a], (0.0, times[-1]), [e0, 0.0], t_eval=times)
    return sol.y[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=("push", "rendering", "half_turn"), default="push")
    parser.add_argument("--inertia", type=float, default=5.0, help="rendering 병진 관성 A_t (논문 0.5, 5, 100)")
    parser.add_argument("--duration", type=float, default=None, help="sim 시간 [s], 없으면 viewer를 닫을 때까지")
    parser.add_argument("--friction", action="store_true", help="MJCF 관절 마찰 10 N·m·감쇠 0.1 유지")
    args = parser.parse_args()

    kernel = MujocoKernel(INDY7_V3_SCENE_XML, INDY7_V3_JOINT_NAMES)
    model, data = kernel.model, kernel.data
    model.opt.timestep = 1.0 / CTRL_HZ
    if not args.friction:
        model.dof_frictionloss[:] = 0.0
        model.dof_damping[:] = 0.0
    kernel.reset_to_keyframe()

    wrapper = PinocchioWrapper(INDY7_V3_URDF, INDY7_V3_FRAME_NAMES)
    wrapper.update_model(kernel.joint_pos, kernel.joint_vel)
    home_pose = wrapper.frame_pose_w("ee").homogeneous.copy()
    pose_des = target_pose(args.experiment, home_pose)

    if args.experiment == "push":
        A, D, K = impedance((2.0, 80.0, 800.0), (0.2, 8.0, 80.0))
    elif args.experiment == "rendering":
        A, D, K = impedance((args.inertia, 20.0, 20.0), (0.5, 10.0, 10.0))
    else:
        A, D, K = impedance((4.0, 150.0, 40.0), (4.0, 150.0, 40.0))
    controller = LieGroupImpedanceController(A, D, K)

    zeros6 = np.zeros(6)
    logger = DataLogger()

    def on_control(t):
        dq = kernel.joint_vel
        wrapper.update_model(kernel.joint_pos, dq)
        jacobian_b = wrapper.jacobian_b("ee")
        pose = wrapper.frame_pose_w("ee").homogeneous
        twist_ref_dot = controller.reference_acceleration(
            pose, jacobian_b @ dq, pose_des, zeros6, zeros6, zeros6, zeros6
        )
        torque = controller.joint_torque(
            twist_ref_dot, jacobian_b, wrapper.jdot_velocity_b("ee"), wrapper.mass_matrix, wrapper.bias_forces, zeros6
        )
        kernel.apply_joint_torque(torque)
        logger.log(times=t, lam=controller.lam, torque=torque, pos=pose[:3, 3])

    def on_render(t, visualizer):
        pose = wrapper.frame_pose_w("ee")
        visualizer.draw_axes(pose.translation, pose.rotation, size=0.15)
        visualizer.draw_axes(pose_des[:3, 3], pose_des[:3, :3], size=0.1)

    duration = args.duration or (None if args.experiment == "push" else 40.0)
    SimScheduler(model, data, ctrl_hz=CTRL_HZ).run(on_control, on_render, duration=duration)

    times, lam, torque = logger.get("times"), logger.get("lam"), logger.get("torque")
    torque_limit = model.actuator_ctrlrange[:, 1]
    print(f"max |τ| / limit: {np.abs(torque).max(axis=0) / torque_limit}")
    print(f"final |η| {np.linalg.norm(lam[-1, :3]):.2e} m, |ξ| {np.linalg.norm(lam[-1, 3:]):.2e} rad")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[1].plot(times, np.linalg.norm(lam[:, 3:], axis=1), label="|ξ| sim")
    if args.experiment == "rendering":
        # 저감쇠 overshoot 부호 유지 → 처음 오차 방향 투영
        direction = lam[0, :3] / np.linalg.norm(lam[0, :3])
        error = lam[:, :3] @ direction
        ideal = ideal_response(args.inertia, D[0, 0], K[0, 0], error[0], times)
        axes[0].plot(times, error, label="η · η₀/|η₀| sim")
        axes[0].plot(times, ideal, "--", label="ideal A ë + D ė + K e = 0")
        print(
            f"η vs ideal: max {np.abs(error - ideal).max():.2e} m, RMS {np.sqrt(((error - ideal) ** 2).mean()):.2e} m"
        )
    else:
        axes[0].plot(times, np.linalg.norm(lam[:, :3], axis=1), label="|η| sim")
    axes[0].set_ylabel("position error [m]")
    axes[1].set_ylabel("orientation error |ξ| [rad]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.suptitle(f"Indy7 v3 Lie group impedance: {args.experiment}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
