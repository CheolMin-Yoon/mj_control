import numpy as np
from scipy.linalg import expm

# 비선형 시스템 이산화

def euler(f, x, u, dt):
    return x + dt * f(x, u)

def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# 선형 시스템 이산화

def zoh(A, B, dt):
    """ZOH: 정확한 이산화. (A_d, B_d) = (e^{A·dt}, A^{-1}(A_d - I)B) — expm으로 한 번에."""
    n = A.shape[0]
    M = np.block([[A, B], [np.zeros((B.shape[1], n + B.shape[1]))]])
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]

def taylor1(A, B, dt):
    return np.eye(A.shape[0]) + A * dt, B * dt

def taylor2(A, B, dt):
    """2차 테일러: A_d ≈ I + A·dt + (A·dt)²/2, B_d ≈ (I·dt + A·dt²/2)·B."""
    n = A.shape[0]
    I = np.eye(n)
    A_d = I + A*dt + 0.5 * (A @ A) * dt**2
    B_d = (I*dt + 0.5 * A * dt**2) @ B
    return A_d, B_d

