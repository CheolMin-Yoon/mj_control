"""Cartpole 해석 동역학, upright 평형점 선형화"""

import numpy as np
from numpy.linalg import inv


class CartpoleDynamics:
    """질점 pole cartpole, 상태 x = [cart_pos, pole_angle, cart_vel, pole_vel], pole_angle 0 = upright

    인자:
        m_c: cart 질량 [kg]
        m_p: pole 질량 [kg]
        L: pole 길이 [m]
    """

    def __init__(self, m_c=1.0, m_p=1.0, L=1.0):
        self.m_c = m_c
        self.m_p = m_p
        self.L = L
        self.g = 9.81

    def continuous_dynamics(self, x, u):
        """dx/dt (4,), u는 cart 힘 [N]"""
        _, pole_angle, cart_vel, pole_vel = x
        dq = np.array([[cart_vel], [pole_vel]])
        B = np.array([[1], [0]])

        M = self._mass_matrix(pole_angle)
        ddq = inv(M) @ (self._gravity_vector(pole_angle) - self._coriolis_matrix(pole_angle, pole_vel) @ dq + B * u)
        return np.array([cart_vel, pole_vel, ddq[0, 0], ddq[1, 0]])

    def linearize(self, x_eq, u_eq):
        """평형점 연속시간 (A (4, 4), B (4, 1)) 반환, B는 힘 [N] 기준"""
        _, pole_angle, _, _ = x_eq

        M0 = self._mass_matrix(pole_angle)
        dGdq = np.array(
            [
                [0, 0],
                [0, self.m_p * self.L * self.g * np.cos(pole_angle)],
            ]
        )
        A = np.block(
            [
                [np.zeros((2, 2)), np.eye(2)],
                [inv(M0) @ dGdq, np.zeros((2, 2))],
            ]
        )
        B = np.vstack([np.zeros((2, 1)), inv(M0) @ np.array([[1], [0]])])
        return A, B

    def _mass_matrix(self, theta):
        return np.array(
            [
                [self.m_c + self.m_p, self.m_p * self.L * np.cos(theta)],
                [self.m_p * self.L * np.cos(theta), self.m_p * self.L**2],
            ]
        )

    def _coriolis_matrix(self, theta, dtheta):
        return np.array(
            [
                [0, -self.m_p * self.L * dtheta**2 * np.sin(theta)],
                [0, 0],
            ]
        )

    def _gravity_vector(self, theta):
        return np.array(
            [
                [0],
                [self.m_p * self.L * self.g * np.sin(theta)],
            ]
        )
