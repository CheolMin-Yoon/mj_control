"""UR5e DLS IK를 OSQP QP로 푸는 데모, 관절 속도 한계와 1-step 관절 위치 한계 제약"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.control import EETracking, TaskSpaceController, TrajectoryGenerator, rectangle_xz_waypoints  # noqa: E402
from source.dynamics import PinocchioWrapper  # noqa: E402
from source.robots.ur5e.constants import UR5E_FRAME_NAMES, UR5E_JOINT_NAMES, UR5E_SCENE_XML, UR5E_URDF  # noqa: E402
from source.sim import DataLogger, MujocoKernel, SimScheduler, hold_until_all_fig_closed, plot_tracking_report  # noqa: E402
from source.solvers import OSQPSolver  # noqa: E402

CTRL_HZ = 100


def main():
    kernel = MujocoKernel(UR5E_SCENE_XML, UR5E_JOINT_NAMES)
    kernel.reset_to_keyframe()
    wrapper = PinocchioWrapper(UR5E_URDF, UR5E_FRAME_NAMES)
    wrapper.update_model(kernel.joint_pos, kernel.joint_vel)
    controller = TaskSpaceController(wrapper.nv)

    init_pose_w = wrapper.frame_pose_w("ee")
    print(f"초기 EE 위치 [m]: {init_pose_w.translation}")
    waypoint_pos_w = rectangle_xz_waypoints(init_pose_w.translation)
    traj = TrajectoryGenerator(waypoint_pos_w, [init_pose_w.rotation.copy()] * len(waypoint_pos_w), target_speed=0.05)

    joint_pos_lower = wrapper.model.lowerPositionLimit
    joint_pos_upper = wrapper.model.upperPositionLimit
    max_joint_vel = wrapper.model.velocityLimit
    nv = wrapper.nv
    solver = OSQPSolver()
    # JᵀJ + λ²I는 dense → placeholder도 dense pattern, 변수 dq에 box 제약
    solver.setup(np.ones((nv, nv)), np.zeros(nv), np.eye(nv), -max_joint_vel, max_joint_vel)

    kernel.reset_to_keyframe()
    tracking = EETracking(kernel, wrapper, controller, traj)
    logger = DataLogger()
    dt = 1.0 / CTRL_HZ
    num_steps = 0

    def on_control(t):
        nonlocal num_steps
        pose_err, _ = tracking.sense(t)
        jacobian_w = wrapper.jacobian_w("ee")
        H, linear_cost = controller.dls_qp_cost(pose_err, tracking.ee_twist_des_w, jacobian_w)

        joint_pos = kernel.joint_pos
        lower = np.maximum(-max_joint_vel, (joint_pos_lower - joint_pos) / dt)
        upper = np.minimum(max_joint_vel, (joint_pos_upper - joint_pos) / dt)
        solver.update(H, linear_cost, lower, upper)
        joint_vel_des = solver.solve()
        if joint_vel_des is None:
            joint_vel_des = np.zeros(nv)

        # 진단: 100 step마다 QP 해와 제약 없는 DLS 해 비교
        if num_steps % 100 == 0:
            joint_vel_dls = controller.dls_joint_vel(pose_err, tracking.ee_twist_des_w, jacobian_w)
            print(
                f"t={t:5.2f}  ||dq_qp - dq_dls||={np.linalg.norm(joint_vel_des - joint_vel_dls):.3e}  "
                f"||dq_qp||={np.linalg.norm(joint_vel_des):.3f}  "
                f"active_lo={int((joint_vel_dls < lower + 1e-9).sum())} "
                f"active_hi={int((joint_vel_dls > upper - 1e-9).sum())}"
            )
        num_steps += 1

        kernel.apply_joint_pos(joint_pos + joint_vel_des * dt)
        kernel.apply_joint_force(wrapper.bias_forces)
        tracking.log(logger, t)

    def on_render(t, visualizer):
        tracking.on_render(t, visualizer)
        visualizer.draw_path(waypoint_pos_w, rgba=(0, 1, 0, 0.5), thickness=0.003)
        visualizer.draw_points(waypoint_pos_w, radius=0.005, rgba=(1, 0, 0, 1))

    SimScheduler(kernel.model, kernel.data, ctrl_hz=CTRL_HZ).run(on_control, on_render)

    plot_tracking_report(logger, waypoint_pos_w, UR5E_JOINT_NAMES, title="UR5e QP", plot_labels=("X", "Z"))
    hold_until_all_fig_closed()


if __name__ == "__main__":
    main()
