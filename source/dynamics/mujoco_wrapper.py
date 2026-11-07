"""MuJoCo API 기구학·동역학, PinocchioWrapper와 같은 interface, Pinocchio 의존 없음"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(slots=True)
class FramePose:
    """pin.SE3와 같은 `translation`·`rotation` 속성의 world 자세"""

    translation: np.ndarray
    rotation: np.ndarray


class MujocoWrapper:
    """kernel model·data 공유, 관절은 kernel 제어 관절 순서, 나머지 관절(그리퍼 등)은 고정값으로 취급

    인자:
        kernel: MujocoKernel
        frame_names: {key: body 또는 site 이름}, body 이름 우선
    """

    def __init__(self, kernel, frame_names):
        self.model = kernel.model
        self.data = kernel.data
        self._qpos_ids = kernel.qpos_ids
        self._dof_ids = kernel.dof_ids
        self.nv = len(self._dof_ids)

        self._frames = {}
        for key, name in frame_names.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._frames[key] = ("body", body_id) if body_id >= 0 else ("site", self.model.site(name).id)

        self._mass_matrix_full = np.zeros((self.model.nv, self.model.nv))
        self._mass_matrix_inv = np.zeros((self.nv, self.nv))
        self._bias_forces = np.zeros(self.nv)
        self.update_model(self.data.qpos[self._qpos_ids], self.data.qvel[self._dof_ids])

    def update_model(self, q, dq):
        """제어 관절 q·dq를 data에 기록 뒤 FK·M·bias force 갱신"""
        self.data.qpos[self._qpos_ids] = q
        self.data.qvel[self._dof_ids] = dq
        self._dq = np.array(dq, copy=True)

        # mj_fwdPosition = kinematics + comPos + crb → 질량행렬 유효
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_fullM(self.model, self.data, self._mass_matrix_full)
        np.copyto(self._mass_matrix_inv, np.linalg.inv(self._mass_matrix_full[np.ix_(self._dof_ids, self._dof_ids)]))

        mujoco.mj_fwdVelocity(self.model, self.data)
        np.copyto(self._bias_forces, self.data.qfrc_bias[self._dof_ids])

    @property
    def mass_matrix_inv(self):
        """제어 관절 블록 M의 역 (nv, nv)"""
        return self._mass_matrix_inv

    @property
    def bias_forces(self):
        """C(q, dq)·dq + g(q) (nv,)"""
        return self._bias_forces

    def frame_pose_w(self, key):
        frame_type, frame_id = self._frames[key]
        if frame_type == "body":
            return FramePose(self.data.xpos[frame_id].copy(), self.data.xmat[frame_id].reshape(3, 3).copy())
        return FramePose(self.data.site_xpos[frame_id].copy(), self.data.site_xmat[frame_id].reshape(3, 3).copy())

    def jacobian_w(self, key):
        """frame 원점 world 축 Jacobian (6, nv), [linear; angular]"""
        frame_type, frame_id = self._frames[key]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if frame_type == "body":
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, frame_id)
        else:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, frame_id)
        return np.vstack([jacp, jacr])[:, self._dof_ids]

    def frame_state_w(self, key):
        """(자세 FramePose, 선속도 (3,), 각속도 (3,)) 반환, 속도는 frame 원점 world 축"""
        twist = self.jacobian_w(key) @ self._dq
        return self.frame_pose_w(key), twist[:3], twist[3:]
