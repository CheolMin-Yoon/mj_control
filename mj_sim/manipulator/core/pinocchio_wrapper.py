'''
Mujoco Kernel로부터 q, dq를 받아 Pinocchio 기반 기구학·동역학을 계산해 노출하는 wrapper.

명명 컨벤션 (MIT/IIT 합성):
- 좌표계: _W (world), _B (body). 단일 대문자 suffix.
- 회전: R_WB (B→W), R_BW (W→B).
- 단일문자: m (mass), M (mass matrix), J (Jacobian), R (rotation).

wrapper._q, wrapper._dq : Pinocchio API 호출 인자 (computeAllTerms 등)
wrapper.q, wrapper.dq   : 외부 read-only 노출
'''


import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Manipulator frame — URDF link
MANIPULATOR_FRAMES = {
    "ee": "tool0",
    "base": "base_link",
}


class Pinocchio_Wrapper:

    def __init__(self, model_path: str, package_dirs=None, frames: dict = None,
                 load_geometry: bool = True):
        """
        model_path    : .urdf 또는 .xml (MJCF) 경로 — 확장자로 자동 분기
        package_dirs  : URDF의 mesh package_dirs (load_geometry=True일 때만)
        frames        : {"ee": "<link_name>", "base": "<link_name>"} 오버라이드.
                        None이면 MANIPULATOR_FRAMES (UR5e 기본).
        load_geometry : URDF visual/collision mesh 로드 여부.
                        False면 kinematic+dynamic tree만 — mesh 경로 깨진 URDF에 사용.
                        MJCF는 geometry 미지원 (자동 무시).
        """
        path = str(model_path)
        if path.endswith('.urdf'):
            if load_geometry:
                robot = RobotWrapper.BuildFromURDF(
                    path, package_dirs=[str(package_dirs)], root_joint=None,
                )
                self.model  = robot.model
                self.vmodel = robot.visual_model
                self.cmodel = robot.collision_model
            else:
                self.model  = pin.buildModelFromUrdf(path)
                self.vmodel = None
                self.cmodel = None
        elif path.endswith('.xml'):
            # MJCF — inertial 정보 풍부, geometry는 미지원
            self.model  = pin.buildModelFromMJCF(path)
            self.vmodel = None
            self.cmodel = None
        else:
            raise ValueError(f"지원 안 되는 모델 형식: {path} (.urdf 또는 .xml만)")

        self.data = self.model.createData()
        self.nv   = self.model.nv
        self.na   = self.model.nv
        self.m    = pin.computeTotalMass(self.model)

        # frame id 캐시 — robot마다 link name 다름 (panda: panda_link0/8, ur5e: base_link/tool0)
        frames = frames if frames is not None else MANIPULATOR_FRAMES
        self.fid = {k: self.model.getFrameId(v) for k, v in frames.items()}

        # 초기 상태
        self.q_init  = pin.neutral(self.model)
        self.dq_init = np.zeros(self.nv)

        # 마지막에 호출 (FK + frame + 동역학 모두 갱신)
        self.update_model(self.q_init, self.dq_init)


    # 매 step 갱신
    def update_model(self, q, dq):
        self._q  = q
        self._dq = dq

        # Pinocchio 4.0 computeAllTerms: FK(joint), M, nle, com, vcom, Jcom,
        # Ag/dAg/hg/dhg, Ig, J, dJ, energy. 단 oMf는 별도 갱신 필요.
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        self._M_inv = pin.computeMinverse(self.model, self.data, q)

        # SE3
        self.oMb   = self.data.oMf[self.fid["base"]]
        self.oM_ee = self.data.oMf[self.fid["ee"]]

        # SO3 — 양방향 모두 캐싱 (호출자 transpose 비용 제거)
        self.R_WB = self.oMb.rotation
        self.R_BW = self.R_WB.T


    # 상태
    @property
    def q(self):  return self._q
    @property
    def dq(self): return self._dq

    # 동역학
    @property
    def M(self):     return self.data.M
    @property
    def M_inv(self): return self._M_inv
    @property
    def C(self):     return pin.computeCoriolisMatrix(self.model, self.data, self._q, self._dq)
    @property
    def nle(self):   return self.data.nle
    @property
    def g(self):     return self.data.g

    # 기구학 (base / com)
    @property
    def p_base_W(self): return self.oMb.translation
    @property
    def p_com_W(self):  return self.data.com[0]
    @property
    def v_com_W(self):  return self.data.vcom[0]
    @property
    def J_com(self):    return self.data.Jcom


    # Jacobian
    def compute_J_W(self, key: str):
        """world-aligned Jacobian (6, nv)."""
        return pin.getFrameJacobian(self.model, self.data,
                                    self.fid[key], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def compute_J_B(self, key: str):
        """body Jacobian (6, nv)."""
        return pin.getFrameJacobian(self.model, self.data,
                                    self.fid[key], pin.ReferenceFrame.LOCAL)

    def compute_Jdot_dq_W(self, key: str):
        """Jdot @ dq (6,) — frame acceleration의 kinematic 부분 (world-aligned)."""
        Jd = pin.getFrameJacobianTimeVariation(self.model, self.data,
                                               self.fid[key], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return Jd @ self._dq


    # 프레임 조회
    def compute_ee_state_W(self, key: str):
        """frame의 SE3 + 선/각속도 (world-aligned)."""
        fid = self.fid[key]
        oMf = self.data.oMf[fid]
        twist = pin.getFrameVelocity(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return oMf, twist.linear, twist.angular


    # 좌표 변환
    def compute_pos_W_to_B(self, pos_W):
        """world 좌표계 pos → base local frame pos."""
        return self.R_BW @ (pos_W - self.oMb.translation)

    def compute_traj_W_to_B(self, traj_W):
        """(N, 3) world 궤적 → base frame 궤적."""
        return (traj_W - self.oMb.translation) @ self.R_BW.T
