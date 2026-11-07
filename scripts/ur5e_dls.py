"""UR5e DLS 속도 IK 데모, position actuator + bias force feedforward off → on 두 번 실행해 비교"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.control import EETracking, TaskSpaceController, TrajectoryGenerator, rectangle_xz_waypoints  # noqa: E402
from source.dynamics import PinocchioWrapper  # noqa: E402
from source.robots.ur5e.constants import UR5E_FRAME_NAMES, UR5E_JOINT_NAMES, UR5E_SCENE_XML, UR5E_URDF  # noqa: E402
from source.sim import DataLogger, MujocoKernel, SimScheduler, hold_until_all_fig_closed, plot_tracking_report  # noqa: E402

CTRL_HZ = 200


def run_once(kernel, wrapper, controller, waypoint_pos_w, waypoint_mat_w, use_bias_feedforward):
    kernel.reset_to_keyframe()
    traj = TrajectoryGenerator(waypoint_pos_w, waypoint_mat_w, target_speed=0.05)
    tracking = EETracking(kernel, wrapper, controller, traj)
    logger = DataLogger()
    dt = 1.0 / CTRL_HZ

    def on_control(t):
        pose_err, _ = tracking.sense(t)
        joint_vel_des = controller.dls_joint_vel(pose_err, tracking.ee_twist_des_w, wrapper.jacobian_w("ee"))
        # 다음 제어 주기까지 1-step 적분 → 위치 명령
        kernel.apply_joint_pos(kernel.joint_pos + joint_vel_des * dt)
        kernel.apply_joint_force(wrapper.bias_forces if use_bias_feedforward else 0.0)
        tracking.log(logger, t)

    def on_render(t, visualizer):
        tracking.on_render(t, visualizer)
        visualizer.draw_path(waypoint_pos_w, rgba=(0, 1, 0, 0.5), thickness=0.003)
        visualizer.draw_points(waypoint_pos_w, radius=0.005, rgba=(1, 0, 0, 1))

    SimScheduler(kernel.model, kernel.data, ctrl_hz=CTRL_HZ).run(on_control, on_render)
    return logger


def main():
    kernel = MujocoKernel(UR5E_SCENE_XML, UR5E_JOINT_NAMES)
    kernel.reset_to_keyframe()
    wrapper = PinocchioWrapper(UR5E_URDF, UR5E_FRAME_NAMES)
    wrapper.update_model(kernel.joint_pos, kernel.joint_vel)
    controller = TaskSpaceController(wrapper.nv)

    init_pose_w = wrapper.frame_pose_w("ee")
    print(f"초기 EE 위치 [m]: {init_pose_w.translation}")
    waypoint_pos_w = rectangle_xz_waypoints(init_pose_w.translation)
    waypoint_mat_w = [init_pose_w.rotation.copy()] * len(waypoint_pos_w)

    # viewer 창을 닫으면 다음 실행
    logger_off = run_once(kernel, wrapper, controller, waypoint_pos_w, waypoint_mat_w, use_bias_feedforward=False)
    logger_on = run_once(kernel, wrapper, controller, waypoint_pos_w, waypoint_mat_w, use_bias_feedforward=True)

    plot_tracking_report(logger_on, waypoint_pos_w, UR5E_JOINT_NAMES, title="UR5e DLS", plot_labels=("X", "Z"))

    plt.figure(figsize=(8, 4.5))
    for logger, style, tag in ((logger_off, "--", "no bias ff"), (logger_on, "-", "bias ff")):
        err_mm = (logger.get("ee_pos_des_w") - logger.get("ee_pos_w")) * 1000.0
        plt.plot(logger.get("times"), err_mm[:, 0], color="red", linestyle=style, label=f"X error ({tag})")
        plt.plot(logger.get("times"), err_mm[:, 2], color="blue", linestyle=style, label=f"Z error ({tag})")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [mm]")
    plt.title("UR5e DLS: bias feedforward off vs on")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    hold_until_all_fig_closed()


if __name__ == "__main__":
    main()
