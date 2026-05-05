import numpy as np
import pinocchio as pin


class FloatingBaseRobotState:

    # G1: base 7(pos3+quat4) + leg 6+6 + waist 3 + arm 7+7 = 36
    # base 6(lin3+ang3) + 위와 동일 = 35
    NQ = 36
    NV = 35

    def __init__(self):
        self._q  = np.zeros(self.NQ)
        self._dq = np.zeros(self.NV)

        # 위치 (q 분할)
        self.humanoid_base_pos  = self._q[0:3]    # pos 3
        self.humanoid_base_quad = self._q[3:7]    # quat 4
        self.left_leg_angle     = self._q[7:13]   # 6
        self.right_leg_angle    = self._q[13:19]  # 6
        self.waist_angle        = self._q[19:22]  # 3
        self.left_arm_angle     = self._q[22:29]  # 7
        self.right_arm_angle    = self._q[29:36]  # 7

        # 속도 (dq 분할)
        self.humanoid_base_lin_vel = self._dq[0:3]   # lin 3
        self.humanoid_base_ang_vel = self._dq[3:6]   # ang 3
        self.left_leg_vel          = self._dq[6:12]  # 6
        self.right_leg_vel         = self._dq[12:18] # 6
        self.waist_vel             = self._dq[18:21] # 3
        self.left_arm_vel          = self._dq[21:28] # 7
        self.right_arm_vel         = self._dq[28:35] # 7

        # yaw unwrap state (compute_rpy_world 첫 호출 시 lazy init)
        self._yaw_unwrap_init = False


    # 상태 (전체)
    @property
    def q(self): return self._q
    @q.setter
    def q(self, q_update): np.copyto(self._q, q_update)

    @property
    def dq(self): return self._dq
    @dq.setter
    def dq(self, dq_update): np.copyto(self._dq, dq_update)


    # 자세 (RPY world, yaw 연속화)
    def compute_rpy_world(self) -> np.ndarray:
        qx, qy, qz, qw = self.humanoid_base_quad
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        roll, pitch, yaw_meas = np.asarray(pin.rpy.matrixToRpy(R)).reshape(3)

        if not self._yaw_unwrap_init:
            self._yaw_unwrap_init = True
            self._yaw_prev_meas   = yaw_meas
            self._yaw_cont        = yaw_meas
        else:
            yaw_delta            = (yaw_meas - self._yaw_prev_meas + np.pi) % (2 * np.pi) - np.pi
            self._yaw_cont      += yaw_delta
            self._yaw_prev_meas  = yaw_meas

        return np.array([roll, pitch, self._yaw_cont])

    def reset_yaw_unwrap(self):
        self._yaw_unwrap_init = False
