import numpy as np
import pinocchio as pin

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Pinocchio_Wrapper

'''
연속 상태방정식 모델을 2차 근사로 이산화하는 클래스

'''


class TrajOptimizer():
    def __init__(self, wrapper: 'Pinocchio_Wrapper', leg: str):
        self.wrapper = wrapper
        self.leg = leg
        
        # 연속 모델
        self.A_c = np.zeros((13, 13))
        self.B_c = np.zeros((13, 12))
        
        self.I_inv = np.linalg.inv(self.wrapper.inertia_matrix)
        self.mass = self.wrapper.mass
        
        self.A_c[3:6, 9:12] = np.eye(3)
        self.A_c[9:12, 12]  = np.array([0, 0, -9.81])
        
        # 이산 모델
        self.A_d = np.zeros((13, 13))
        self.B_d = np.zeros((13, 13))
        
        
    def update_continous_dynamics(self):
        
        r = self.wrapper.get_moment_arm_in_world(self.leg)
        self.A_c[0:3, 6:9] = self.wrapper.R_z.T
        
        for i, leg_name in enumerate(self.leg):
            # i번째 발의 모멘트 암 벡터 r 계산
            r = self.wrapper.get_moment_arm_in_world(leg_name)
            r_skew = pin.skew(r)
            
            # i번째 발의 열 인덱스 범위 계산 (예: 0~3, 3~6, 6~9 ...)
            col_start = 3 * i
            col_end = 3 * (i + 1)
            
            # 회전 모멘트 파트: I_inv * [r_i]_x
            self.B_c[6:9, col_start:col_end] = self.I_inv @ r_skew
            
            # 선가속도 파트: 1/m * I_3
            self.B_c[9:12, col_start:col_end] = np.eye(3) / self.mass
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    '''
    
    출력:
      1. des_x_pos, des_y_pos
      2. des_x_vel, des_y_vel
      3. yaw_rate
    
    '''
    def compute_desired_command(self):
        return 0.0, 0.0