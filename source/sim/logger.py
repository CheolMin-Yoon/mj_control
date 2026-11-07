"""control step 신호 기록, 이름별 numpy 배열 반환"""

from collections import defaultdict

import numpy as np


class DataLogger:
    def __init__(self):
        self._buf = defaultdict(list)

    def log(self, **signals):
        """신호마다 값 사본 append"""
        for key, value in signals.items():
            self._buf[key].append(np.array(value, copy=True))

    def get(self, key):
        """(num_steps, ...) 배열 반환"""
        return np.array(self._buf[key])
