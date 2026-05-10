import numpy as np


# trot 기본 위상 — diagonal pair (FL+RR, FR+RL) 동기화
PHASE_OFFSET = np.array([0.5, 0.0, 0.0, 0.5])
HEIGHT_SWING = 0.08


class GaitScheduler:
    def __init__(self, gait_hz: float, duty: float):
        self.gait_period  = 1.0 / gait_hz
        self.gait_duty    = duty
        self.stance_time  = duty * self.gait_period
        self.swing_time   = (1.0 - duty) * self.gait_period

    def get_mask(self, t: float) -> np.ndarray:
        """contact mask (4,) — 1=stance, 0=swing."""
        phases = np.mod(PHASE_OFFSET + t / self.gait_period, 1.0)
        return (phases < self.gait_duty).astype(np.int32)

    def make_swing_traj(self, p0: np.ndarray, pf: np.ndarray, h_sw: float):
        """5차 다항식 + bezier z-bump → eval_at(dt) 클로저 반환."""
        p0 = np.array(p0, dtype=float)
        pf = np.array(pf, dtype=float)
        dp = pf - p0
        T  = self.swing_time

        def eval_at(dt: float):
            s    = np.clip(dt / T, 0.0, 1.0)
            poly  = 10*s**3 - 15*s**4 + 6*s**5
            dpoly = (30*s**2 - 60*s**3 + 30*s**4) / T
            d2poly= (60*s   - 180*s**2 + 120*s**3) / T**2
            p = p0 + dp * poly
            v = dp * dpoly
            a = dp * d2poly
            if h_sw > 0.0:
                b   =  64*s**3*(1-s)**3
                db  = (192*s**2*(1-s)**2*(1-2*s)) / T
                d2b = (192*(2*s*(1-s)**2*(1-2*s)
                           - 2*s**2*(1-s)*(1-2*s)
                           - 2*s**2*(1-s)**2)) / T**2
                p[2] += h_sw * b
                v[2] += h_sw * db
                a[2] += h_sw * d2b
            return p, v, a

        return eval_at
