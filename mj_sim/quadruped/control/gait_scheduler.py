import numpy as np


# 기본 trot phase — diagonal pair (FL+RR, FR+RL) 동기화. (FL, FR, RL, RR)
PHASE_OFFSET_TROT = np.array([0.5, 0.0, 0.0, 0.5])


class GaitScheduler:
    def __init__(self, gait_hz: float, duty: float, phase_offset=PHASE_OFFSET_TROT):
        assert 0.0 < duty < 1.0, "duty는 (0, 1) 범위 (전부 stance/swing은 별도 처리)"
        self.gait_period  = 1.0 / gait_hz
        self.gait_duty    = duty
        self.stance_time  = duty * self.gait_period
        self.swing_time   = (1.0 - duty) * self.gait_period
        self.phase_offset = np.asarray(phase_offset)

    def compute_gait_state(self, t: float):
        """t → (contact_mask (4,) int 1=stance/0=swing, swing_phases (4,) ∈ [0,1])."""
        phases = np.mod(t / self.gait_period + self.phase_offset, 1.0)
        contact_mask = (phases < self.gait_duty).astype(np.int32)
        swing_phases = np.where(
            contact_mask == 0,
            (phases - self.gait_duty) / (1.0 - self.gait_duty),
            0.0,
        )
        return contact_mask, swing_phases

    def compute_mask_horizon(self, t: float, dt: float, N: int) -> np.ndarray:
        """contact mask horizon (N, 4) — MPC 예측용."""
        ts = t + dt * np.arange(N)
        phases = np.mod(self.phase_offset[None, :] + ts[:, None] / self.gait_period, 1.0)
        return (phases < self.gait_duty).astype(np.int32)
