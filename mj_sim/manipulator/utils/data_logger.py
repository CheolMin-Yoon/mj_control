import numpy as np
from collections import deque


class DataLogger:
    """
    이름 붙인 시그널을 매 control step에 append하고,
    plot_helpers에 바로 넘길 수 있는 numpy 배열로 반환.

    max_steps=None : 무제한 (list 누적, 메모리 주의)
    max_steps=N    : 최근 N step만 (deque, 무한 시뮬용)
    """
    def __init__(self, max_steps=None):
        self.max_steps = max_steps
        self._buf   = {}    # key -> list 또는 deque(maxlen)
        self._cache = {}    # key -> 마지막 get() 결과 (재변환 회피)

    def log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._buf:
                self._buf[k] = (deque(maxlen=self.max_steps)
                                if self.max_steps else [])
            self._buf[k].append(np.asarray(v).copy())
            self._cache.pop(k, None)

    def get(self, key):
        if key in self._cache:
            return self._cache[key]
        arr = np.array(self._buf[key])
        self._cache[key] = arr
        return arr

    def keys(self):
        return list(self._buf.keys())

    def reset(self):
        self._buf.clear()
        self._cache.clear()

    def __getitem__(self, key):
        return self.get(key)
