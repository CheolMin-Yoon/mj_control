"""궤적 추종 데모 공통 단계: 상태 읽기 → dynamics 갱신 → 목표 평가 → 현재 frame 상태 → 오차"""

import numpy as np


class EETracking:
    """한 frame 궤적 추종의 감지·기록·overlay 담당, 명령 계산은 호출자 몫

    인자:
        kernel: MujocoKernel
        wrapper: PinocchioWrapper 또는 MujocoWrapper, kernel 제어 관절 순서
        controller: TaskSpaceController
        traj: TrajectoryGenerator
        frame: wrapper frame key
    """

    def __init__(self, kernel, wrapper, controller, traj, frame="ee"):
        self.kernel = kernel
        self.wrapper = wrapper
        self.controller = controller
        self.traj = traj
        self.frame = frame

        self.ee_pos_des_w = np.zeros(3)
        self.ee_mat_des_w = np.eye(3)
        self.ee_twist_des_w = np.zeros(6)
        self.ee_pos_w = np.zeros(3)
        self.ee_mat_w = np.eye(3)
        self.ee_twist_w = np.zeros(6)

    def sense(self, t):
        """(pose_err (6,), twist_err (6,)) 반환, wrapper는 현재 q·dq로 갱신된 상태"""
        self.wrapper.update_model(self.kernel.joint_pos, self.kernel.joint_vel)

        pose_des_w, twist_des_w = self.traj.evaluate(t)
        self.ee_pos_des_w[:] = pose_des_w.translation
        self.ee_mat_des_w[:] = pose_des_w.rotation
        self.ee_twist_des_w[:] = twist_des_w

        pose_w, lin_vel_w, ang_vel_w = self.wrapper.frame_state_w(self.frame)
        self.ee_pos_w[:] = pose_w.translation
        self.ee_mat_w[:] = pose_w.rotation
        self.ee_twist_w[:3] = lin_vel_w
        self.ee_twist_w[3:] = ang_vel_w

        return self.controller.tracking_error(
            self.ee_pos_w,
            self.ee_mat_w,
            self.ee_twist_w,
            self.ee_pos_des_w,
            self.ee_mat_des_w,
            self.ee_twist_des_w,
        )

    def log(self, logger, t):
        """`plot_tracking_report` 신호 기록"""
        logger.log(
            times=t,
            ee_pos_des_w=self.ee_pos_des_w,
            ee_pos_w=self.ee_pos_w,
            ee_lin_vel_des_w=self.ee_twist_des_w[:3],
            ee_lin_vel_w=self.ee_twist_w[:3],
            q=self.kernel.joint_pos,
            dq=self.kernel.joint_vel,
            ddq=self.kernel.joint_acc,
        )

    def on_render(self, t, visualizer):
        visualizer.draw_axes(self.ee_pos_w, self.ee_mat_w, size=0.2)
        visualizer.draw_axes(self.ee_pos_des_w, self.ee_mat_des_w, size=0.2)
