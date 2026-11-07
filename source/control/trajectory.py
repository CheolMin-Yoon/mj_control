"""waypoint 사이 정지-정지 quintic 궤적"""

import numpy as np
import pinocchio as pin


class TrajectoryGenerator:
    """구간마다 위치는 직선, 자세는 log3 축, 둘 다 s(τ) = 10τ³ − 15τ⁴ + 6τ⁵로 보간

    waypoint마다 속도·가속도 0, 구간 시간 = 거리 / target_speed(최소 5 ms)

    인자:
        waypoint_pos_w: (N, 3) 위치 [m]
        waypoint_mat_w: (N, 3, 3) 회전행렬
        target_speed: 구간 평균 속도 [m/s]
    """

    def __init__(self, waypoint_pos_w, waypoint_mat_w, target_speed):
        self.waypoint_pos_w = np.asarray(waypoint_pos_w)
        self.waypoint_mat_w = np.asarray(waypoint_mat_w)
        segment_lengths = np.linalg.norm(np.diff(self.waypoint_pos_w, axis=0), axis=1)
        self.waypoint_times = np.concatenate([[0.0], np.cumsum(np.maximum(segment_lengths / target_speed, 0.005))])

    @property
    def duration(self):
        return self.waypoint_times[-1]

    def evaluate(self, t):
        """(목표 자세 pin.SE3, 목표 twist (6,) [linear; angular] world 축) 반환"""
        if t >= self.duration:
            return pin.SE3(self.waypoint_mat_w[-1], self.waypoint_pos_w[-1]), np.zeros(6)

        i = int(np.searchsorted(self.waypoint_times, t, side="right")) - 1
        T = self.waypoint_times[i + 1] - self.waypoint_times[i]
        tau = (t - self.waypoint_times[i]) / T
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T

        delta_pos = self.waypoint_pos_w[i + 1] - self.waypoint_pos_w[i]
        mat_0 = self.waypoint_mat_w[i]
        rot_vec = pin.log3(self.waypoint_mat_w[i + 1] @ mat_0.T)
        pos = self.waypoint_pos_w[i] + delta_pos * s
        mat = pin.exp3(rot_vec * s) @ mat_0
        return pin.SE3(mat, pos), np.concatenate([delta_pos * ds, rot_vec * ds])


def rectangle_xz_waypoints(start_pos_w, width=0.25, height=0.25):
    """start에서 +x로 width, −z로 height인 x-z 평면 사각형, 시작점으로 복귀 (5, 3)"""
    offsets = np.array([[0, 0, 0], [width, 0, 0], [width, 0, -height], [0, 0, -height], [0, 0, 0]])
    return np.asarray(start_pos_w) + offsets
