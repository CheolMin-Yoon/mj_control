'''
Mujoco Kernel로부터 q, dq를 받아 MuJoCo API 기반 기구학·동역학을 계산해 노출하는 wrapper.
Pinocchio_Wrapper와 동일한 인터페이스를 제공한다.

명명 컨벤션:
- 좌표계: _W (world), _B (body).
- 회전: R_WB (B→W), R_BW (W→B).

MuJoCo API 대응:
- FK            : mj_fwdPosition
- Jacobian      : mj_jacBody / mj_jacSite  (arm 열만 슬라이싱: _col)
- 질량 행렬 M   : mj_fullM  (arm×arm 블록)
- nle (C@dq+g)  : data.qfrc_bias[:nv]  (mj_fwdVelocity 이후)
- g             : dq=0 상태에서 data.qfrc_bias[:nv]
- M_inv         : np.linalg.inv(M)

nv : arm DOF — kernel._tau_perm 길이 (그리퍼 등 비-arm DOF 제외)
_col : arm actuator의 MuJoCo qvel 인덱스 배열 (= kernel._tau_perm)

frames 키는 MuJoCo body 이름 또는 site 이름으로 전달.
body 이름이면 mj_jacBody, site 이름이면 mj_jacSite를 자동 선택.
'''


import numpy as np
import mujoco


MANIPULATOR_FRAMES = {
    "ee":   "tool0",
    "base": "base_link",
}


def _make_full_M(model, data):
    M_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    return M_full


class Mujoco_Wrapper:

    def __init__(self, kernel, frames: dict = None):
        """
        kernel : Mujoco_Kernel 인스턴스 — model/data를 참조로 공유 (복사 없음)
        frames : {"ee": "<body_or_site_name>", "base": "<body_or_site_name>"}
                 None이면 MANIPULATOR_FRAMES (UR5e 기본).
        """
        self.model = kernel.model
        self.data  = kernel.data

        # arm DOF — kernel._tau_perm이 arm actuator의 MuJoCo qvel 인덱스
        self._col = kernel._tau_perm          # shape (na,)  — qvel 열 인덱스
        self.nv   = len(self._col)            # arm DOF (pinocchio 기준)
        self.na   = self.nv

        frames = frames if frames is not None else MANIPULATOR_FRAMES

        # body id / site id 캐시 — body 우선, 없으면 site
        self._fid   = {}
        self._ftype = {}   # "body" | "site"
        for key, name in frames.items():
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._fid[key]   = bid
                self._ftype[key] = "body"
            else:
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid < 0:
                    raise ValueError(f"'{name}' 을 body 또는 site에서 찾을 수 없습니다.")
                self._fid[key]   = sid
                self._ftype[key] = "site"

        # 총 질량
        self.m = float(np.sum(self.model.body_mass))

        # 내부 버퍼 (arm DOF 기준)
        self._nle_buf = np.zeros(self.nv)
        self._g_buf   = np.zeros(self.nv)
        self._M_buf   = np.zeros((self.nv, self.nv))
        self._M_inv   = np.zeros((self.nv, self.nv))

        # kernel의 현재 qpos/qvel (keyframe reset 이후 호출 가정)
        q0  = self.data.qpos[self._col].copy()
        dq0 = self.data.qvel[self._col].copy()
        self.update_model(q0, dq0)


    def update_model(self, q, dq):
        self._q  = q
        self._dq = dq

        # arm 관절만 MuJoCo state에 반영 (그리퍼 등 나머지는 건드리지 않음)
        self.data.qpos[self._col] = q
        self.data.qvel[self._col] = dq

        # FK + 관성 텐서 (mj_fwdPosition = kinematics + comPos + crb)
        mujoco.mj_fwdPosition(self.model, self.data)

        # 질량 행렬 — arm×arm 블록만 추출
        M_full = _make_full_M(self.model, self.data)
        np.copyto(self._M_buf, M_full[np.ix_(self._col, self._col)])
        np.copyto(self._M_inv, np.linalg.inv(self._M_buf))

        # nle = C@dq + g  (arm DOF만)
        mujoco.mj_fwdVelocity(self.model, self.data)
        np.copyto(self._nle_buf, self.data.qfrc_bias[self._col])

        # g: dq=0에서 qfrc_bias (arm만 0으로)
        self.data.qvel[self._col] = 0.0
        mujoco.mj_fwdVelocity(self.model, self.data)
        np.copyto(self._g_buf, self.data.qfrc_bias[self._col])
        self.data.qvel[self._col] = dq   # 복원

        # base/ee frame SE3
        self._update_frame_cache()


    def _update_frame_cache(self):
        bid = self._fid["base"]
        self.R_WB = self.data.xmat[bid].reshape(3, 3).copy()
        self.R_BW = self.R_WB.T
        self._p_base_W = self.data.xpos[bid].copy()

        ee_id = self._fid["ee"]
        if self._ftype["ee"] == "body":
            self._p_ee_W = self.data.xpos[ee_id].copy()
            self._R_ee_W = self.data.xmat[ee_id].reshape(3, 3).copy()
        else:
            self._p_ee_W = self.data.site_xpos[ee_id].copy()
            self._R_ee_W = self.data.site_xmat[ee_id].reshape(3, 3).copy()


    # 상태
    @property
    def q(self):  return self._q
    @property
    def dq(self): return self._dq

    # 동역학 (모두 arm DOF 기준)
    @property
    def M(self):     return self._M_buf
    @property
    def M_inv(self): return self._M_inv
    @property
    def nle(self):   return self._nle_buf
    @property
    def g(self):     return self._g_buf
    @property
    def C(self):
        """Coriolis 행렬 (nv, nv) — 열별 perturbation. 비용 큼."""
        eps = 1e-7
        nle0 = self._nle_buf - self._g_buf
        C = np.zeros((self.nv, self.nv))
        dq_pert = self._dq.copy()
        for i in range(self.nv):
            dq_pert[i] += eps
            self.data.qvel[self._col] = dq_pert
            mujoco.mj_fwdVelocity(self.model, self.data)
            C[:, i] = (self.data.qfrc_bias[self._col] - self._g_buf - nle0) / eps
            dq_pert[i] -= eps
        self.data.qvel[self._col] = self._dq   # 복원
        mujoco.mj_fwdVelocity(self.model, self.data)
        return C

    # 기구학
    @property
    def p_base_W(self): return self._p_base_W
    @property
    def p_com_W(self):  return self.data.subtree_com[0].copy()
    @property
    def v_com_W(self):  return self.J_com[:3] @ self._dq
    @property
    def J_com(self):
        """COM Jacobian (3, nv) — arm 열만."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jacp, 0)
        return jacp[:, self._col]


    # Jacobian
    def compute_J_W(self, key: str):
        """world-aligned Jacobian (6, nv) — arm 열만."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        fid  = self._fid[key]
        if self._ftype[key] == "body":
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, fid)
        else:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, fid)
        J_full = np.vstack([jacp, jacr])
        return J_full[:, self._col]

    def compute_J_B(self, key: str):
        """body-frame Jacobian (6, nv)."""
        J_W = self.compute_J_W(key)
        R = self.R_BW if key == "base" else self._R_ee_W.T
        T = np.block([[R, np.zeros((3, 3))],
                      [np.zeros((3, 3)), R]])
        return T @ J_W

    def compute_Jdot_dq_W(self, key: str):
        """Jdot @ dq (6,) — 수치 미분."""
        eps = 1e-7
        J0  = self.compute_J_W(key)
        self.data.qpos[self._col] += eps * self._dq
        mujoco.mj_fwdPosition(self.model, self.data)
        J1  = self.compute_J_W(key)
        self.data.qpos[self._col] -= eps * self._dq
        mujoco.mj_fwdPosition(self.model, self.data)
        return ((J1 - J0) / eps) @ self._dq


    # 프레임 조회
    def compute_ee_state_W(self, key: str):
        """frame SE3 + 선속도 + 각속도 (world-aligned)."""
        fid = self._fid[key]
        if self._ftype[key] == "body":
            p = self.data.xpos[fid].copy()
            R = self.data.xmat[fid].reshape(3, 3).copy()
        else:
            p = self.data.site_xpos[fid].copy()
            R = self.data.site_xmat[fid].reshape(3, 3).copy()

        J = self.compute_J_W(key)
        twist = J @ self._dq
        return _SE3(p, R), twist[:3], twist[3:]


    # 좌표 변환
    def compute_pos_W_to_B(self, pos_W):
        return self.R_BW @ (pos_W - self._p_base_W)

    def compute_traj_W_to_B(self, traj_W):
        return (traj_W - self._p_base_W) @ self.R_BW.T


class _SE3:
    """Pinocchio SE3와 동일한 .translation / .rotation 인터페이스."""
    __slots__ = ("translation", "rotation")
    def __init__(self, p, R):
        self.translation = p
        self.rotation    = R
