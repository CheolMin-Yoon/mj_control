"""MuJoCo viewer user_scn 디버그 overlay"""

import mujoco
import numpy as np


class Visualizer:
    """매 render frame `reset` 뒤 `draw_*` 호출, maxgeom 초과분은 무시

    인자:
        viewer: mujoco.viewer passive handle
    """

    def __init__(self, viewer):
        self.scn = viewer.user_scn
        # mjv_initGeom 기본 인자, mjv_connector가 덮어씀
        self._zero3 = np.zeros(3)
        self._zero9 = np.zeros(9)
        self._eye_flat = np.eye(3).flatten()
        self._axis_rgba = (
            np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        )

    def reset(self):
        self.scn.ngeom = 0

    def draw_axes(self, pos_w, mat_w, size=0.1, shaft=0.005):
        """frame XYZ 화살표, 빨강·초록·파랑"""
        mat_w = np.asarray(mat_w).reshape(3, 3)
        pos_w = np.asarray(pos_w).reshape(3)
        for i in range(3):
            if not self._connector(
                mujoco.mjtGeom.mjGEOM_ARROW, shaft, pos_w, pos_w + mat_w[:, i] * size, self._axis_rgba[i]
            ):
                return

    def draw_points(self, points_w, radius=0.01, rgba=(1.0, 0.5, 0.0, 1.0)):
        """점마다 구, points_w (N, 3) 또는 (3,)"""
        rgba = np.asarray(rgba, dtype=np.float32)
        size = np.full(3, radius)
        for point in np.asarray(points_w).reshape(-1, 3):
            if self.scn.ngeom >= self.scn.maxgeom:
                return
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                point,
                self._eye_flat,
                rgba,
            )
            self.scn.ngeom += 1

    def draw_path(self, points_w, rgba=(0.0, 1.0, 0.0, 1.0), thickness=0.005):
        """연속한 점을 캡슐로 연결"""
        points_w = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
        rgba = np.asarray(rgba, dtype=np.float32)
        for start, end in zip(points_w[:-1], points_w[1:], strict=True):
            if not self._connector(mujoco.mjtGeom.mjGEOM_CAPSULE, thickness, start, end, rgba):
                return

    def _connector(self, geom_type, width, start, end, rgba):
        if self.scn.ngeom >= self.scn.maxgeom:
            return False
        geom = self.scn.geoms[self.scn.ngeom]
        mujoco.mjv_initGeom(geom, geom_type, self._zero3, self._zero3, self._zero9, rgba)
        mujoco.mjv_connector(geom, geom_type, width, start, end)
        self.scn.ngeom += 1
        return True
