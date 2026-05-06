"""
MuJoCo 시뮬레이터와 제어 시스템 간의 통신을 담당하는 최적화된 커널.
상태 업데이트는 FloatingBaseRobotState 객체의 메모리에 직접 덮어씀
"""


import numpy as np
import mujoco
import pinocchio as pin


class Mujoco_Kernel:
    def __init__(self, xml_path, joint_names_pin_order=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        nq = self.model.nq
        nv = self.model.nv
        self._q_buf  = np.zeros(nq)   # full MuJoCo state — state container 호환
        self._dq_buf = np.zeros(nv)

        if not joint_names_pin_order or any(n is Ellipsis for n in joint_names_pin_order):
            raise ValueError("joint_names_pin_order를 명시적으로 전달해야 합니다.")

        # joint-driven actuator만 매핑 (tendon-driven 등 비-joint actuator는 별도 namespace, joint_id 덮어쓰기 방지)
        joint_id_to_act_id = {
            self.model.actuator_trnid[i, 0]: i
            for i in range(self.model.nu)
            if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        }
        try:
            self._tau_perm = np.array([
                joint_id_to_act_id[self.model.joint(name).id]
                for name in joint_names_pin_order
            ], dtype=int)
        except KeyError as e:
            raise ValueError(f"Pinocchio 관절 {e} 가 MuJoCo Actuator에 연결되어 있지 않습니다!")

        # ctrl 버퍼는 perm 길이 — Pinocchio nv와 일치 (그리퍼 등 비-Pinocchio actuator 제외)
        self._tau = np.zeros(len(self._tau_perm))

        # MJCF keyframe[0] 자세 (없으면 None)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.q_keyframe = self.data.qpos.copy()
            mujoco.mj_resetData(self.model, self.data)
        else:
            self.q_keyframe = None


    def reset_to_keyframe(self, key_id: int = 0):
        """MJCF keyframe 자세로 시뮬 초기화."""
        if self.model.nkey > key_id:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)


    # 상태 읽기
    @property
    def q_mj(self):
        return self.data.qpos
    @property
    def dq_mj(self):
        return self.data.qvel
    
    def update_robot_state(self, state):
        q_mj  = self.q_mj
        dq_mj = self.dq_mj
        np.copyto(self._q_buf, q_mj)
        np.copyto(self._dq_buf, dq_mj)
        state.q  = self._q_buf
        state.dq = self._dq_buf

    # 상태 쓰기 — Pinocchio 관절 순서로 (그리퍼 등은 별도: kernel.data.ctrl[idx] 직접)
    @property
    def ctrl_pos(self):
        return self.data.ctrl[self._tau_perm]
    @ctrl_pos.setter
    def ctrl_pos(self, q_des):
        """q_des shape = (len(_tau_perm),) — Pinocchio nv 일치"""
        self.data.ctrl[self._tau_perm] = q_des

    @property
    def ctrl_tau(self):
        return self._tau

    @ctrl_tau.setter
    def ctrl_tau(self, tau_des):
        """tau_des shape = (len(_tau_perm),) — Pinocchio nv 일치"""
        np.copyto(self._tau, tau_des)
        self.data.ctrl[self._tau_perm] = self._tau

    def step(self):
        mujoco.mj_step(self.model, self.data)