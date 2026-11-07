"""UR5e OSC 임피던스 데모, 7개 waypoint의 위치·자세 추종"""

import sys
from pathlib import Path

import numpy as np
from pinocchio import rpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.control import EETracking, TaskSpaceController, TrajectoryGenerator  # noqa: E402
from source.dynamics import PinocchioWrapper  # noqa: E402
from source.robots.ur5e.constants import UR5E_FRAME_NAMES, UR5E_JOINT_NAMES, UR5E_TORQUE_SCENE_XML, UR5E_URDF  # noqa: E402
from source.sim import DataLogger, MujocoKernel, SimScheduler, hold_until_all_fig_closed, plot_tracking_report  # noqa: E402

CTRL_HZ = 500


def main():
    kernel = MujocoKernel(UR5E_TORQUE_SCENE_XML, UR5E_JOINT_NAMES)
    kernel.reset_to_keyframe()
    wrapper = PinocchioWrapper(UR5E_URDF, UR5E_FRAME_NAMES)
    wrapper.update_model(kernel.joint_pos, kernel.joint_vel)
    controller = TaskSpaceController(wrapper.nv)

    init_pose_w = wrapper.frame_pose_w("ee")
    init_pos_w = init_pose_w.translation.copy()
    init_mat_w = init_pose_w.rotation.copy()
    print(f"초기 EE 위치 [m]: {init_pos_w}")

    offsets = [
        [0, 0, 0],
        [-0.2, 0.2, -0.1],
        [0.1, 0, -0.25],
        [0, -0.1, -0.3],
        [-0.1, 0.2, -0.1],
        [-0.2, -0.1, -0.1],
        [0.4, 0.3, 0],
    ]
    rpy_offsets = [
        (0, 0, 0),
        (0, 0, np.pi / 4),
        (0, np.pi / 6, 0),
        (0, 0, np.pi / 2),
        (np.pi / 6, np.pi / 6, np.pi / 2),
        (0, np.pi / 6, 0),
        (np.pi / 6, 0, 0),
    ]
    waypoint_pos_w = init_pos_w + np.array(offsets)
    waypoint_mat_w = [init_mat_w @ rpy.rpyToMatrix(*angles) for angles in rpy_offsets]
    traj = TrajectoryGenerator(waypoint_pos_w, waypoint_mat_w, target_speed=0.1)

    kernel.reset_to_keyframe()
    tracking = EETracking(kernel, wrapper, controller, traj)
    logger = DataLogger()

    def on_control(t):
        pose_err, twist_err = tracking.sense(t)
        kernel.apply_joint_torque(
            controller.impedance_torque(
                pose_err,
                twist_err,
                wrapper.jacobian_w("ee"),
                wrapper.mass_matrix_inv,
                wrapper.bias_forces,
            )
        )
        tracking.log(logger, t)

    SimScheduler(kernel.model, kernel.data, ctrl_hz=CTRL_HZ).run(on_control, tracking.on_render)

    plot_tracking_report(logger, waypoint_pos_w, UR5E_JOINT_NAMES, title="UR5e impedance")
    hold_until_all_fig_closed()


if __name__ == "__main__":
    main()
