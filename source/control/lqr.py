"""연속시간 무한구간 LQR"""

from numpy.linalg import inv
from scipy.linalg import solve_continuous_are


def lqr_gain(A, B, Q, R):
    """CARE 해 P로 상태 feedback gain K = R⁻¹BᵀP 반환, u = −K x"""
    P = solve_continuous_are(A, B, Q, R)
    return inv(R) @ B.T @ P
