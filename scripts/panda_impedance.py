"""Panda OSC 임피던스 데모, 초기 TCP 주변 사각 경로 추종

dynamics는 `--dynamics pinocchio`(기본, wn 10) 또는 `--dynamics mujoco`(MuJoCo API만, wn 20)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.control import EETracking, TaskSpaceController, TrajectoryGenerator  # noqa: E402
from source.dynamics import MujocoWrapper, PinocchioWrapper  # noqa: E402
from source.robots.panda.constants import (  # noqa: E402
    PANDA_GRIPPER_ACTUATOR_ID,
    PANDA_GRIPPER_OPEN_CTRL,
    PANDA_JOINT_NAMES,
    PANDA_MUJOCO_FRAME_NAMES,
    PANDA_PINOCCHIO_FRAME_NAMES,
    PANDA_PINOCCHIO_XML,
    PANDA_SCENE_XML,
)
from source.sim import DataLogger, MujocoKernel, SimScheduler, hold_until_all_fig_closed, plot_tracking_report  # noqa: E402

CTRL_HZ = 500


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics", choices=("pinocchio", "mujoco"), default="pinocchio")
    args = parser.parse_args()

    kernel = MujocoKernel(PANDA_SCENE_XML, PANDA_JOINT_NAMES)
    kernel.reset_to_keyframe()
    if args.dynamics == "pinocchio":
        wrapper = PinocchioWrapper(PANDA_PINOCCHIO_XML, PANDA_PINOCCHIO_FRAME_NAMES)
        wrapper.update_model(kernel.joint_pos, kernel.joint_vel)
        controller = TaskSpaceController(wrapper.nv, wn_pos=10.0, wn_ori=10.0)
    else:
        wrapper = MujocoWrapper(kernel, PANDA_MUJOCO_FRAME_NAMES)
        controller = TaskSpaceController(wrapper.nv, wn_pos=20.0, wn_ori=20.0)

    init_pose_w = wrapper.frame_pose_w("ee")
    init_pos_w = init_pose_w.translation.copy()
    init_mat_w = init_pose_w.rotation.copy()
    print(f"초기 EE 위치 [m]: {init_pos_w}")

    offsets = [[0, 0, 0], [0.10, 0.15, 0], [-0.10, 0.15, -0.10], [-0.10, -0.15, -0.10], [0.10, -0.15, 0], [0, 0, 0]]
    waypoint_pos_w = init_pos_w + np.array(offsets)
    traj = TrajectoryGenerator(waypoint_pos_w, [init_mat_w] * len(waypoint_pos_w), target_speed=0.2)

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
        kernel.data.ctrl[PANDA_GRIPPER_ACTUATOR_ID] = PANDA_GRIPPER_OPEN_CTRL
        tracking.log(logger, t)

    SimScheduler(kernel.model, kernel.data, ctrl_hz=CTRL_HZ).run(on_control, tracking.on_render)

    plot_tracking_report(logger, waypoint_pos_w, PANDA_JOINT_NAMES, title=f"Panda impedance ({args.dynamics})")
    hold_until_all_fig_closed()


if __name__ == "__main__":
    main()
