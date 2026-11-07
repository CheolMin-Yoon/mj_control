"""Pinocchio 고정 기반 기구학·동역학"""

import numpy as np
import pinocchio as pin


class PinocchioWrapper:
    """`update_model` 한 번으로 FK·M⁻¹·bias force·frame Jacobian 갱신

    인자:
        model_path: `.urdf` 또는 MJCF `.xml`, 고정 기반
        frame_names: {key: model frame 이름}
    """

    def __init__(self, model_path, frame_names):
        model_path = str(model_path)
        if model_path.endswith(".urdf"):
            self.model = pin.buildModelFromUrdf(model_path)
        else:
            self.model = pin.buildModelFromMJCF(model_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.frame_ids = {key: self.model.getFrameId(name) for key, name in frame_names.items()}
        self.update_model(pin.neutral(self.model), np.zeros(self.nv))

    def update_model(self, q, dq):
        # computeAllTerms는 oMf 미갱신 → updateFramePlacements 별도
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        self._mass_matrix_inv = pin.computeMinverse(self.model, self.data, q)
        self._dq = dq

    @property
    def mass_matrix_inv(self):
        return self._mass_matrix_inv

    @property
    def bias_forces(self):
        """C(q, dq)·dq + g(q) (nv,)"""
        return self.data.nle

    def frame_pose_w(self, key):
        """world 기준 frame 자세 pin.SE3, data 참조"""
        return self.data.oMf[self.frame_ids[key]]

    def jacobian_w(self, key):
        """frame 원점 world 축 Jacobian (6, nv), [linear; angular]"""
        return pin.getFrameJacobian(self.model, self.data, self.frame_ids[key], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def frame_state_w(self, key):
        """(자세 pin.SE3, 선속도 (3,), 각속도 (3,)) 반환, 속도는 frame 원점 world 축"""
        frame_id = self.frame_ids[key]
        twist = pin.getFrameVelocity(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return self.data.oMf[frame_id], twist.linear, twist.angular
