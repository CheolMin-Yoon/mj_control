import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


# G1 frame — URDF link (foot sole = virtual OP_FRAME added in __init__)
G1_FRAMES = {
    "L_foot": "L_foot_sole",
    "R_foot": "R_foot_sole",
    "L_hand": "left_rubber_hand",
    "R_hand": "right_rubber_hand",
    "L_hip" : "left_hip_pitch_link",
    "R_hip" : "right_hip_pitch_link",
    "L_sh"  : "left_shoulder_pitch_link",
    "R_sh"  : "right_shoulder_pitch_link",
    "base"  : "pelvis",
}

# contact geom centroid in ankle_roll_link local frame
# 4 sphere geoms: [-0.05,±0.025,-0.03] (rear) and [0.12,±0.03,-0.03] (front) → centroid [0.035, 0, -0.03]
FOOT_SOLE_OFFSET = np.array([0.035, 0.0, -0.03])
_ANKLE_LINKS = {"L_foot": "left_ankle_roll_link", "R_foot": "right_ankle_roll_link"}

LEG_KEYS = ("L_foot", "R_foot")
LEG_IDX  = {k: i for i, k in enumerate(LEG_KEYS)}


class Pinocchio_Wrapper:

    def __init__(self, urdf_path: str, package_dirs,
                 leg_keys=LEG_KEYS):
        robot = RobotWrapper.BuildFromURDF(
            str(urdf_path),
            package_dirs = [str(package_dirs)],
            root_joint = pin.JointModelFreeFlyer()
        )

        # 모델
        self.model  = robot.model
        self.vmodel = robot.visual_model
        self.cmodel = robot.collision_model

        # virtual foot sole frames (contact geom centroid offset from ankle_roll_link)
        for key in ("L_foot", "R_foot"):
            ankle_fid = self.model.getFrameId(_ANKLE_LINKS[key])
            ankle_f   = self.model.frames[ankle_fid]
            self.model.addFrame(pin.Frame(
                G1_FRAMES[key],
                ankle_f.parent,
                ankle_fid,
                ankle_f.placement * pin.SE3(np.eye(3), FOOT_SOLE_OFFSET),
                pin.FrameType.OP_FRAME,
            ))
        self.data = self.model.createData()

        # 기본 정보
        self.nv   = self.model.nv
        self.na   = self.model.nv - 6           # actuated DOF (floating base 6 제외)
        self.mass = pin.computeTotalMass(self.model)

        # 프레임 ID
        self.fid = {k: self.model.getFrameId(v) for k, v in G1_FRAMES.items()}

        # leg config
        self.leg_keys = list(leg_keys)
        self.leg_idx  = {k: i for i, k in enumerate(self.leg_keys)}

        # neutral pose에서 hip / shoulder placement 캐싱
        q_neutral = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q_neutral)
        oMb_neutral = self.data.oMf[self.fid["base"]]
        self.hip_placements = {
            k: oMb_neutral.actInv(self.data.oMf[self.fid[f"{k[0]}_hip"]]).copy()
            for k in self.leg_keys
        }
        self.L_shoulder_placement = oMb_neutral.actInv(self.data.oMf[self.fid["L_sh"]]).copy()
        self.R_shoulder_placement = oMb_neutral.actInv(self.data.oMf[self.fid["R_sh"]]).copy()

        # 초기 상태
        self.q_init  = q_neutral.copy()
        self.dq_init = np.zeros(self.nv)

        # 버퍼 (R_z [2,2]=1은 init 후 불변, update_model에서 yaw만 in-place 갱신)
        self.R_z = np.eye(3)

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
        self.oMb      = self.data.oMf[self.fid["base"]]
        self.oM_Lfoot = self.data.oMf[self.fid["L_foot"]]
        self.oM_Rfoot = self.data.oMf[self.fid["R_foot"]]
        self.oM_Lhand = self.data.oMf[self.fid["L_hand"]]
        self.oM_Rhand = self.data.oMf[self.fid["R_hand"]]

        # SO3
        self.R_body_to_world = self.oMb.rotation
        self.R_world_to_body = self.R_body_to_world.T

        # R_z yaw 회전
        yaw = np.arctan2(self.R_body_to_world[1, 0], self.R_body_to_world[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self.R_z[0, 0] =  c; self.R_z[0, 1] = -s
        self.R_z[1, 0] =  s; self.R_z[1, 1] =  c


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
    def base_pos(self):      return self.data.oMf[self.fid["base"]].translation
    @property
    def com_pos_world(self): return self.data.com[0]
    @property
    def com_vel_world(self): return self.data.vcom[0]
    @property
    def J_com(self):         return self.data.Jcom

    # 중심 운동량
    @property
    def hg(self):               return self.data.hg
    @property
    def Ag(self):               return self.data.Ag
    @property
    def angular_momentum(self): return self.data.hg.angular
    @property
    def inertia_matrix(self):   return self.data.Ig.inertia


    # Jacobian
    def get_J_world(self, key: str):
        return pin.getFrameJacobian(self.model, self.data,
                                    self.fid[key], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def get_J_body(self, key: str):
        return pin.getFrameJacobian(self.model, self.data,
                                    self.fid[key], pin.ReferenceFrame.LOCAL)

    def get_Jdot_dq_world(self, key: str):
        Jd = pin.getFrameJacobianTimeVariation(self.model, self.data,
                                               self.fid[key], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return Jd @ self._dq


    # 프레임 조회
    def get_ee_state_world(self, key: str):
        fid = self.fid[key]
        oMf = self.data.oMf[fid]
        twist = pin.getFrameVelocity(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return oMf, twist.linear, twist.angular

    def get_moment_arm_in_world(self, key: str):
        return self.data.oMf[self.fid[key]].translation - self.com_pos_world

    def get_hip_offset(self, leg: str):
        return self.hip_placements[leg].translation


    # 좌표 변환
    def world_to_base_frame(self, pos_world):
        return self.R_world_to_body @ (pos_world - self.oMb.translation)

    def trajectory_world_to_base(self, traj_world):
        return (traj_world - self.oMb.translation) @ self.R_world_to_body.T
