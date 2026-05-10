import numpy as np
import mujoco

# 모듈 상수 — mjv_initGeom의 기본 인자 (mjv_connector로 덮어쓰는 dummy)

_ZERO3        = np.zeros(3)
_ZERO9        = np.zeros(9)
_IDENTITY_MAT = np.eye(3).flatten()

_AXIS_COLORS = (
    np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),  # X 빨강
    np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),  # Y 초록
    np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),  # Z 파랑
)


class Visualizer:
    """
    mujoco viewer.user_scn에 디버그용 시각화 (좌표계, 궤적, 접촉력)을 그린다.
    매 렌더 프레임 시작 시 reset() 호출 후 draw_* 호출.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.scn = viewer.user_scn

    # 매 렌더 프레임 시작에 호출
    def reset(self):
        self.scn.ngeom = 0


    # 좌표계 (XYZ 화살표)
    def draw_axes(self, pos, R, size=0.1, shaft=0.005):
        R = np.asarray(R).reshape(3, 3)
        pos = np.asarray(pos).reshape(3)
        for i in range(3):
            if self.scn.ngeom >= self.scn.maxgeom:
                return
            end = pos + R[:, i] * size
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                _ZERO3, _ZERO3, _ZERO9,
                _AXIS_COLORS[i],
            )
            mujoco.mjv_connector(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW, shaft,
                pos, end,
            )
            self.scn.ngeom += 1

    # 여러 frame 한꺼번에 (oMf_dict: {key: pin.SE3 또는 (pos, R) 튜플})
    def draw_frame_set(self, oMf_dict, size=0.1):
        for _, oMf in oMf_dict.items():
            if hasattr(oMf, 'translation') and hasattr(oMf, 'rotation'):
                self.draw_axes(oMf.translation, oMf.rotation, size=size)
            else:
                pos, R = oMf
                self.draw_axes(pos, R, size=size)


    # 점 시각화 (waypoints/contact points)
    def draw_trajectory(self, waypoints, radius=0.01, rgba=(1.0, 0.5, 0.0, 1.0)):
        wp = np.asarray(waypoints)
        if wp.ndim == 1:
            wp = wp.reshape(1, 3)
        rgba_f = np.asarray(rgba, dtype=np.float32)
        size = np.full(3, radius)
        for p in wp:
            if self.scn.ngeom >= self.scn.maxgeom:
                return
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size, np.asarray(p).reshape(3), _IDENTITY_MAT,
                rgba_f,
            )
            self.scn.ngeom += 1


    # 두 점 잇는 캡슐 (ConvexHull 테두리, 지지선 등)
    def draw_line(self, p1, p2, rgba=(0.0, 1.0, 0.0, 1.0), thickness=0.005):
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            _ZERO3, _ZERO3, _ZERO9,
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE, thickness,
            np.asarray(p1, dtype=np.float64).reshape(3),
            np.asarray(p2, dtype=np.float64).reshape(3),
        )
        self.scn.ngeom += 1

    # waypoint 시퀀스를 잇는 선
    def draw_connected_path(self, waypoints, rgba=(0.0, 1.0, 0.0, 1.0), thickness=0.005):
        wp = np.asarray(waypoints)
        if len(wp) < 2:
            return
        for i in range(len(wp) - 1):
            self.draw_line(wp[i], wp[i + 1], rgba=rgba, thickness=thickness)


    # 접촉력 (접촉점 sphere + force arrow)
    def draw_contact_force(self, p, f, scale=0.005, rgba=(1.0, 1.0, 0.0, 1.0),
                           shaft=0.006, point_radius=0.008):
        if self.scn.ngeom + 2 > self.scn.maxgeom:
            return
        p = np.asarray(p).reshape(3)
        f = np.asarray(f).reshape(3)
        rgba_f = np.asarray(rgba, dtype=np.float32)

        # 접촉점 구체
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, point_radius), p, _IDENTITY_MAT, rgba_f,
        )
        self.scn.ngeom += 1

        # 힘 화살표
        end = p + f * scale
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            _ZERO3, _ZERO3, _ZERO9, rgba_f,
        )
        mujoco.mjv_connector(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW, shaft, p, end,
        )
        self.scn.ngeom += 1


    # 임의 geom 직접 추가 (size/pos/mat/rgba 모두 명시)
    def draw_geom(self, geom_type, size, pos, mat, rgba):
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            geom_type,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.asarray(mat, dtype=np.float64).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        self.scn.ngeom += 1
