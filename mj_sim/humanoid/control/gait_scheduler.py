import numpy as np


class GaitScheduler:

    def __init__(self, gait_hz: float, duty: float, phase_offset=(0.0, 0.5)):
        assert 0.0 < duty < 1.0, "duty는 (0, 1) 범위여야 함 (전부 stance/swing 케이스 별도 처리)"
        self.gait_period  = 1.0 / gait_hz
        self.gait_duty    = duty
        self.stance_time  = duty * self.gait_period
        self.swing_time   = (1.0 - duty) * self.gait_period
        self.phase_offset = np.asarray(phase_offset)


    # 매 step 갱신 (시간 t에서 각 다리의 contact / swing phase 계산)
    def compute_gait_state(self, t: float):
        """t (sec) → (contact_mask (n_legs,) int, swing_phases (n_legs,) ∈ [0,1])."""
        phases       = np.mod(t / self.gait_period + self.phase_offset, 1.0)
        contact_mask = (phases < self.gait_duty).astype(int)
        swing_phases = np.where(
            contact_mask == 0,
            (phases - self.gait_duty) / (1.0 - self.gait_duty),
            0.0,
        )
        return contact_mask, swing_phases
