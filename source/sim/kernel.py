"""MuJoCo model·data 소유, 제어 관절 순서의 qpos·dof·actuator 주소 매핑"""

import mujoco


class MujocoKernel:
    """시뮬레이터 상태 읽기와 관절 명령 쓰기 담당

    인자:
        xml_path: 장면 MJCF 경로
        joint_names: 제어 관절 이름, 이 순서가 `joint_pos`·명령·dynamics wrapper의 관절 순서
    """

    def __init__(self, xml_path, joint_names):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        joint_ids = [self.model.joint(name).id for name in joint_names]
        actuator_ids_by_joint = {
            self.model.actuator_trnid[i, 0]: i
            for i in range(self.model.nu)
            if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        }
        self.actuator_ids = [actuator_ids_by_joint[joint_id] for joint_id in joint_ids]
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.dof_ids = self.model.jnt_dofadr[joint_ids]

        # keyframe 0 자세, 없으면 None
        self.keyframe_qpos = None
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.keyframe_qpos = self.data.qpos.copy()
            mujoco.mj_resetData(self.model, self.data)

    def reset_to_keyframe(self):
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)

    @property
    def qpos(self):
        return self.data.qpos

    @property
    def qvel(self):
        return self.data.qvel

    @property
    def joint_pos(self):
        """제어 관절 위치 사본 (num_joints,)"""
        return self.data.qpos[self.qpos_ids]

    @property
    def joint_vel(self):
        """제어 관절 속도 사본 (num_joints,)"""
        return self.data.qvel[self.dof_ids]

    @property
    def joint_acc(self):
        """제어 관절 가속도 사본 (num_joints,)"""
        return self.data.qacc[self.dof_ids]

    def apply_joint_torque(self, joint_torque):
        """motor actuator ctrl에 관절 순서 torque 기록"""
        self.data.ctrl[self.actuator_ids] = joint_torque

    def apply_joint_pos(self, joint_pos_des):
        """position actuator ctrl에 관절 순서 목표 위치 기록"""
        self.data.ctrl[self.actuator_ids] = joint_pos_des

    def apply_joint_force(self, joint_force):
        """qfrc_applied 전체 0 뒤 제어 관절 dof에 일반화 힘 기록"""
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof_ids] = joint_force
