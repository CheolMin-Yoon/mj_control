import numpy as np
import pinocchio as pin

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Pinocchio_Wrapper


class SingleRigidBodyDynamics:
    def __init__(self, wrapper: Pinocchio_Wrapper, legs: list):
        self.wrapper = wrapper
        self.legs    = legs
        self.n_legs  = len(legs)
        self.A_c = np.zeros((13, 13))
        self.B_c = np.zeros((13, 3 * self.n_legs))
        
    
    
    def continuous_system(self):
        
        self.A_c[:3, 6:9] = self.wrapper.R_yaw
        self.A_c[3:6, 9:12] = np.eye(3)
        self.A_c[11, 13] = - self.g
        
        self.B_c[6:9, 0:12] = np.linalg.inv(self.wrapper.I_world) @ pin.skew(self.wrapper.compute_moment_arm_W)
        self.B_c[9:12, 0:12] = np.array([self.wrapper.m * np.eye(3), self.wrapper.m * np.eye(3), self.wrapper.m * np.eye(3), self.wrapper.m * np.eye(3)])
        
        return self.A_c, self.B_c
    
    
    def 