import numpy as np
import scipy.linalg as la
import pinocchio as pin


class SingleRigidBodyDynamics:
    def __init__(self, wrapper, legs: list, MPC_dt: float, N: int):
        self.wrapper = wrapper
        self.legs    = legs
        self.n_legs  = len(legs)
        
        self.A_c = np.zeros((13, 13))
        self.B_c = np.zeros((13, 3 * self.n_legs))
        self.A_d = np.eye(13)
        self.B_d = np.zeros((13, 3 * self.n_legs))
        self.A_arr = np.zeros((N, 13, 13))
        self.B_arr = np.zeros((N, 13, 12))
        self.A_qp = np.zeros((13 * N, 13))
        self.B_qp = np.zeros((13 * N, 12 * N)) 
        self.L = np.zeros((13 * N, 13 * N))
        self.K = np.zeros((12 * N, 12 * N))

        self.N = N
        self.dt = MPC_dt
        self.m = self.wrapper.m
        self.g = 9.81
        self.I_inv_B = np.linalg.inv(self.wrapper.I_B)
        self.R_yaw = self.wrapper.R_WH
        self.r_W = lambda i: self.wrapper.compute_moment_arm_W(self.legs[i])
    
    # 식 13~17
    def continuous_system(self):
        
        self.A_c[0:3, 6:9]  = self.R_yaw
        self.A_c[3:6, 9:12] = np.eye(3)
        self.A_c[11,  12]   = self.g
        
        B_omega_blocks = []
        for i in range(self.n_legs):
            skew_r = pin.skew(self.r_W(i))
            B_omega_blocks.append(self.I_inv_B @ skew_r)
        B_v_block = (1.0 / self.m) * np.eye(3)

        self.B_c[6:9, 0:12]  = np.hstack(B_omega_blocks)
        self.B_c[9:12, 0:12] = np.hstack([B_v_block] * self.n_legs)
        
        return self.A_c, self.B_c
    
    # 식 25~26
    def discrete_system(self):
        
        self.A_d[0:3, 6:9]   = self.R_yaw * self.dt
        self.A_d[3:6, 9:12]  = np.eye(3) * self.dt
        self.A_d[5,   12]    = 0.5 * self.g * (self.dt**2) 
        self.A_d[11,  12]    = self.g * self.dt

        B_omega_blocks = []
        for i in range(self.n_legs):
            skew_r = pin.skew(self.r_W(i))
            B_omega_blocks.append(self.I_inv_B @ skew_r)
        B_omega_matrix = np.hstack(B_omega_blocks)

        B_p_block = (0.5 * (self.dt ** 2) / self.m) * np.eye(3)
        B_v_block = (self.dt / self.m) * np.eye(3)

        self.B_d[0:3, 0:12]  = 0.5 * (self.R_yaw @ B_omega_matrix) * (self.dt**2)
        self.B_d[3:6, 0:12] = np.hstack([B_p_block] * self.n_legs)
        self.B_d[6:9, 0:12] = B_omega_matrix * self.dt
        self.B_d[9:12, 0:12] = np.hstack([B_v_block] * self.n_legs)

        return self.A_d, self.B_d
    
    # 식 27~32 
    def quadratic_form_system(self, x_0, x_des, Q, R):

        self.L = la.block_diag(*[Q for _ in range(self.N)])
        self.K = la.block_diag(*[R for _ in range(self.N)])

        A_acc = np.eye(13)
        
        # 2. A_qp 와 B_qp 블록 행렬 조립 <-- 여기는 어렵다...
        for i in range(self.N):
            # A_qp의 i번째 행 블록 조립
            A_acc = self.A_arr[i] @ A_acc
            self.A_qp[i*13 : (i+1)*13, :] = A_acc
            
            # B_qp의 대각 블록 채우기 (현재 스텝의 입력 B_i)
            self.B_qp[i*13 : (i+1)*13, i*12 : (i+1)*12] = self.B_arr[i]
            
            # B_qp의 하삼각 블록 채우기 (과거 입력의 전파)
            if i > 0:
                # 이전 스텝(i-1) 행의 채워진 부분(0 ~ i*nu)을 통째로 떼어옴
                prev_row_blocks = self.B_qp[(i-1)*13 : i*13, 0 : i*12]
                
                # A_d_list[i] 한 번의 행렬 곱으로 전체 블록을 업데이트하여 삽입!
                self.B_qp[i*13 : (i+1)*13, 0 : i*12] = self.A_arr[i] @ prev_row_blocks

        H = 2 * (self.B_qp.T @ self.L @ self.B_qp + self.K)
        g = 2 * self.B_qp.T @ self.L @ (self.A_qp @ x_0 - x_des)

        return H, g