import numpy as np
import scipy.linalg as la


class SingleRigidBodyDynamics:
    def __init__(self, wrapper, Q, R, MPC_dt: float, N: int):
        self.wrapper = wrapper

        self.A_c = np.zeros((13, 13))
        self.B_c = np.zeros((13, 12))
        self.A_d = np.eye(13)
        self.B_d = np.zeros((13, 12))
        self.A_arr = np.zeros((N, 13, 13))
        self.B_arr = np.zeros((N, 13, 12))
        self.A_qp = np.zeros((13 * N, 13))
        self.B_qp = np.zeros((13 * N, 12 * N))
        self.L = la.block_diag(*[Q] * N)          # (13 * N, 13 * N)
        self.K = la.block_diag(*[R] * N)          # (12 * N, 12 * N)
        self.H = np.zeros((12*N, 12*N))
        self.q = np.zeros(12*N)

        self._I3_tile = np.tile(np.eye(3), 4)
        self._I3 = np.eye(3)
        self._r_skew_buf = np.zeros((3, 12))   # (3, 12) — 4 leg skew(r) hstack
        self.N = N
        self.dt = MPC_dt
        self.m = self.wrapper.m
        self.g = 9.81
        # R_HW, I_inv_W는 매 tick 갱신 (R_WH·yaw 변동, I_W 시간 변동 반영)


    # 식 13~17
    def compute_AB_c(self, contact_mask):
        grf_mask = np.repeat(contact_mask, 3)
        R_HW    = self.wrapper.R_WH.T
        I_inv_W = np.linalg.inv(self.wrapper.I_W)

        self.A_c[0:3, 6:9]  = R_HW
        self.A_c[3:6, 9:12] = self._I3
        self.A_c[11,  12]   = self.g

        self.B_c[6:9,  0:12] = (I_inv_W @ self.wrapper.r_skew_W) * grf_mask
        self.B_c[9:12, 0:12] = (self._I3_tile / self.m) * grf_mask
        return self.A_c, self.B_c

    # 식 25~26 — r_feet_lever: (3, 4) leg lever arms (foot - base) at this horizon step
    def compute_AB_d(self, contact_mask, r_feet_lever=None):
        grf_mask = np.repeat(contact_mask, 3)
        dt = self.dt
        R_HW    = self.wrapper.R_WH.T            # rpy_dot = R_HW · omega_W (reference 동일)
        I_inv_W = np.linalg.inv(self.wrapper.I_W)  # world frame inertia inverse

        self.A_d[0:3, 6:9]  = R_HW * dt
        self.A_d[3:6, 9:12] = self._I3 * dt
        self.A_d[5,   12]   = 0.5 * self.g * dt**2
        self.A_d[11,  12]   = self.g * dt

        # r_skew: r_feet_lever 주어지면 그 시점 lever로, 아니면 현재 wrapper값 사용
        if r_feet_lever is None:
            r_skew = self.wrapper.r_skew_W
        else:
            for i in range(4):
                r = r_feet_lever[:, i]
                self._r_skew_buf[:, 3*i:3*i+3] = np.array([
                    [    0, -r[2],  r[1]],
                    [ r[2],     0, -r[0]],
                    [-r[1],  r[0],     0],
                ])
            r_skew = self._r_skew_buf

        B_omega = (I_inv_W @ r_skew) * grf_mask
        B_v     = (self._I3_tile / self.m) * grf_mask

        self.B_d[0:3,  0:12] = 0.5 * (R_HW @ B_omega) * dt**2
        self.B_d[3:6,  0:12] = 0.5 * B_v * dt**2
        self.B_d[6:9,  0:12] = B_omega * dt
        self.B_d[9:12, 0:12] = B_v * dt
        return self.A_d, self.B_d

    # horizon N step LTV 모델 stack
    # r_feet_horizon: (3, 4, N) leg lever arm trajectory. None이면 현재 시점 한 번만 사용.
    def update_horizon(self, mask_horizon, r_feet_horizon=None):
        for k in range(self.N):
            r_k = None if r_feet_horizon is None else r_feet_horizon[:, :, k]
            self.compute_AB_d(mask_horizon[k], r_k)
            self.A_arr[k] = self.A_d
            self.B_arr[k] = self.B_d

    # 식 27~32
    def compute_condensed_qp(self, x_0, x_des):
        A_acc = np.eye(13)

        for i in range(self.N):
            A_acc = self.A_arr[i] @ A_acc
            self.A_qp[i*13 : (i+1)*13, :] = A_acc
            self.B_qp[i*13 : (i+1)*13, i*12 : (i+1)*12] = self.B_arr[i]
            if i > 0:
                self.B_qp[i*13 : (i+1)*13, 0 : i*12] = (
                    self.A_arr[i] @ self.B_qp[(i-1)*13 : i*13, 0 : i*12]
                )

        self.H[:] = 2 * (self.B_qp.T @ self.L @ self.B_qp + self.K)
        self.q[:] = 2 * self.B_qp.T @ self.L @ (self.A_qp @ x_0 - x_des)

        return self.H, self.q