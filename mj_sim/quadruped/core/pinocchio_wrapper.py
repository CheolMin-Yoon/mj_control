'''
Mujoco Kernel로부터 q, dq를 받아 Pinocchio 기반 기구학·동역학을 계산해 노출하는 wrapper.

명명 컨벤션 (MIT/IIT 합성):
- 좌표계: _W (world), _B (body), _H (hip/horizontal). 단일 대문자 suffix.
- 회전: R_WB (B→W), R_BW (W→B), R_WH (yaw-only, H=horizontal, SRBD-MPC용).
- 단일문자: m (mass), M (mass matrix), J (Jacobian), R (rotation), I_body / I_world (inertia).
- 발 인덱싱: FL/FR/RL/RR. r_feet은 (3, 4) 컬럼 순서 [FL, FR, RL, RR].

wrapper._q, wrapper._dq : Pinocchio API 호출 인자 (computeAllTerms 등)
wrapper.q, wrapper.dq   : 외부 read-only 노출
'''


import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Go2 frame — URDF link
GO2_FRAMES = {
    "FL": "FL_foot",
    "FR": "FR_foot",
    "RL": "RL_foot",
    "RR": "RR_foot",
    "FL_hip": "FL_thigh_joint",
    "FR_hip": "FR_thigh_joint",
    "RL_hip": "RL_thigh_joint",
    "RR_hip": "RR_thigh_joint",
    "base":   "base",
}

LEG_KEYS = ("FL", "FR", "RL", "RR")
LEG_IDX  = {k: i for i, k in enumerate(LEG_KEYS)}


class Pinocchio_Wrapper:

    def __init__(self, urdf_path: str, package_dirs):
        robot = RobotWrapper.BuildFromURDF(
            str(urdf_path),
            package_dirs=[str(package_dirs)],
            root_joint=pin.JointModelFreeFlyer(),
        )

        self.model  = robot.model
        self.vmodel = robot.visual_model
        self.cmodel = robot.collision_model
        self.data   = self.model.createData()
        self.nv    = self.model.nv
        self.na   = self.model.nv - 6
        self.m     = pin.computeTotalMass(self.model)

        # body-frame inertia of base (URDF inertial 그대로) — SRBD-MPC용. 시간 불변.
        # joint 0 = universe, 1 = freeflyer (root)의 base body
        self.I_B = self.model.inertias[1].inertia.copy()

        # frame id 캐시
        self.fid = {k: self.model.getFrameId(v) for k, v in GO2_FRAMES.items()}

        # neutral pose에서 4발 hip placement (base_T_root) 캐싱
        q_neutral = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q_neutral)
        oMb_neutral = self.data.oMf[self.fid["base"]]
        self.hip_placements = {
            k: oMb_neutral.actInv(self.data.oMf[self.fid[f"{k}_hip"]]).copy()
            for k in LEG_KEYS
        }

        # 초기 상태
        self.q_init  = q_neutral.copy()
        self.dq_init = np.zeros(self.nv)

        # 버퍼 (R_WH [2,2]=1은 init 후 불변, update_model에서 yaw만 in-place 갱신)
        self.R_WH  = np.eye(3)
        self._r_feet = np.zeros((3, 4))   # [FL, FR, RL, RR] columns
        
        self._r_feet_com_W   = np.zeros((3, 4))
        self._r_skew_packed  = np.zeros((3, 12))
        self._r_skew_views   = tuple(self._r_skew_packed[:, 3*i:3*i+3] for i in range(4))

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

        # SE3 — base + 4발
        self.oMb    = self.data.oMf[self.fid["base"]]
        self.oM_FL  = self.data.oMf[self.fid["FL"]]
        self.oM_FR  = self.data.oMf[self.fid["FR"]]
        self.oM_RL  = self.data.oMf[self.fid["RL"]]
        self.oM_RR  = self.data.oMf[self.fid["RR"]]

        # SO3 — 양방향 모두 캐싱 (호출자 transpose 비용 제거)
        self.R_WB = self.oMb.rotation
        self.R_BW = self.R_WB.T

        # R_WH — yaw-only 회전 (SRBD-MPC 선형화에 사용)
        yaw = np.arctan2(self.R_WB[1, 0], self.R_WB[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self.R_WH[0, 0] =  c; self.R_WH[0, 1] = -s
        self.R_WH[1, 0] =  s; self.R_WH[1, 1] =  c

        # r_feet — (3, 4) 4발 위치 world frame, 컬럼 순서 [FL, FR, RL, RR]
        self._r_feet[:, 0] = self.oM_FL.translation
        self._r_feet[:, 1] = self.oM_FR.translation
        self._r_feet[:, 2] = self.oM_RL.translation
        self._r_feet[:, 3] = self.oM_RR.translation

        # r_feet_com_W — (3, 4) 발-CoM, world frame
        np.subtract(self._r_feet, self.p_com_W[:, None], out=self._r_feet_com_W)

        # r_skew_W — (3, 12) 각 다리 skew(r_i)를 hstack한 형태, _r_skew_views로 in-place 채움
        r = self._r_feet_com_W
        for i in range(4):
            s = self._r_skew_views[i]
            rx, ry, rz = r[0, i], r[1, i], r[2, i]
            s[0, 1] = -rz; s[0, 2] =  ry
            s[1, 0] =  rz; s[1, 2] = -rx
            s[2, 0] = -ry; s[2, 1] =  rx


    # 상태
    @property
    def q(self):  return self._q
    @property
    def dq(self): return self._dq
    @property
    def r_feet_com_W(self):  return self._r_feet_com_W
    @property
    def r_skew_W(self):      return self._r_skew_packed   
    
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
    @property
    def I_W(self):   return self.data.Ig.inertia

    # 기구학
    @property
    def p_base_W(self):  return self.oMb.translation
    @property
    def p_com_W(self):   return self.data.com[0]
    @property
    def v_com_W(self):   return self.data.vcom[0]
    @property
    def J_com(self):     return self.data.Jcom
    @property
    def r_feet(self):
        """(3, 4) 4발 위치 world frame, 컬럼 순서 [FL, FR, RL, RR]."""
        return self._r_feet

    # 운동량
    @property
    def hg(self): return self.data.hg
    @property
    def Ag(self): return self.data.Ag


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

    def compute_moment_arm_W(self, key: str):
        """key 프레임의 CoM 기준 모멘트 암 (world)."""
        return self.data.oMf[self.fid[key]].translation - self.p_com_W

    def get_hip_offset(self, leg: str):
        """leg ('FL'|'FR'|'RL'|'RR') 의 hip placement translation (base frame)."""
        return self.hip_placements[leg].translation


    # 좌표 변환
    def compute_pos_W_to_B(self, pos_W):
        """world 좌표계 pos → base local frame pos."""
        return self.R_BW @ (pos_W - self.oMb.translation)

    def compute_traj_W_to_B(self, traj_W):
        """(N, 3) world 궤적 → base frame 궤적."""
        return (traj_W - self.oMb.translation) @ self.R_BW.T
