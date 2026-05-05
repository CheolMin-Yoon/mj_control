from collections import deque
import numpy as np


class DataLogger:
    """
    이름 붙인 시그널을 매 control step에 append. plot_helpers로 바로 넘길 수 있는 numpy 배열로 반환.

    max_steps 지정 시 ring buffer (가장 오래된 것 drop). None이면 무제한 (단기 디버그용).
    각 시그널은 첫 append shape으로 고정 — 이후 mismatch 시 ValueError.
    """

    def __init__(self, max_steps: int = None):
        self.max_steps = max_steps
        self._buf      = {}        # key -> deque(maxlen) or list
        self._shape    = {}        # key -> 첫 append shape (고정)


    def log(self, **kwargs):
        for k, v in kwargs.items():
            arr = np.asarray(v)

            # 첫 append: shape 고정 + buffer 생성
            if k not in self._buf:
                self._shape[k] = arr.shape
                self._buf[k]   = (deque(maxlen=self.max_steps)
                                  if self.max_steps is not None else [])
            elif arr.shape != self._shape[k]:
                raise ValueError(
                    f"DataLogger['{k}'] shape mismatch: 첫 append {self._shape[k]}, 현재 {arr.shape}"
                )

            self._buf[k].append(arr.copy())


    def get(self, key):
        return np.array(self._buf[key])


    def reset(self):
        self._buf.clear()
        self._shape.clear()


    def __getitem__(self, key):
        return self.get(key)
