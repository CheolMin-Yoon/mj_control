r"""
$$
\begin{aligned}
V&=\begin{bmatrix}v\\ \omega\end{bmatrix},\qquad \lambda=\begin{bmatrix}\eta\\ \xi\end{bmatrix} \\
\theta&=\lVert\xi\rVert \\
\theta^2<1&:\ \text{급수},\qquad \theta^2\ge1:\ \text{닫힌 해}\quad(\text{FP32 상쇄 오차 방지})
\end{aligned}
$$
"""

from math import factorial

import casadi as ca

# 급수 계수, θ^{2j} 계수 (j = 0, 1, …)
_NUM_TERMS = 8
_BERNOULLI = (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6)


def _series(theta2, coefs):
    r"""
    $$
    \sum_{j}c_j\,\theta^{2j}\ (\text{Horner})
    $$
    """
    out = coefs[-1]
    for coef in reversed(coefs[:-1]):
        out = out * theta2 + coef
    return out


def _coefs():
    r"""
    $$
    \begin{aligned}
    \alpha&=\sum_{j\ge0}\frac{(-1)^j\theta^{2j}}{(2j+1)!},\qquad
    \beta=\sum_{j\ge0}\frac{2(-1)^j\theta^{2j}}{(2j+2)!} \\
    \Gamma_1&=\sum_{j\ge0}\frac{(-1)^j\theta^{2j}}{(2j+3)!},\qquad
    \Gamma_2=\sum_{j\ge0}(-1)^{j+1}\Big(\frac{1}{(2j+3)!}-\frac{2}{(2j+4)!}\Big)\theta^{2j} \\
    \Gamma_3&=\sum_{j\ge0}(-1)^{j+1}\Big(\frac{1}{(2j+4)!}-\frac{3}{(2j+5)!}\Big)\theta^{2j},\qquad
    \Gamma_4=\sum_{n\ge1}\frac{-(-1)^nB_{2n}}{(2n)!}\theta^{2n-2}
    \end{aligned}
    $$
    """
    j = range(_NUM_TERMS)
    return {
        "alpha": [(-1) ** k / factorial(2 * k + 1) for k in j],
        "beta": [2 * (-1) ** k / factorial(2 * k + 2) for k in j],
        "gamma1": [(-1) ** k / factorial(2 * k + 3) for k in j],
        "gamma2": [(-1) ** (k + 1) * (1 / factorial(2 * k + 3) - 2 / factorial(2 * k + 4)) for k in j],
        "gamma3": [(-1) ** (k + 1) * (1 / factorial(2 * k + 4) - 3 / factorial(2 * k + 5)) for k in j],
        "gamma4": [-((-1) ** n) * b / factorial(2 * n) for n, b in enumerate(_BERNOULLI, start=1)],
    }


_COEFS = _coefs()


def _scalar_coefs(xi):
    r"""
    $$
    \begin{aligned}
    \alpha&=\frac{\sin\theta}{\theta},\qquad \beta=\frac{2(1-\cos\theta)}{\theta^2},\qquad
    \gamma=\frac{\alpha}{\beta}=\frac{\theta}{2}\cot\frac{\theta}{2} \\
    \Gamma_1&=\frac{1-\alpha}{\theta^2},\qquad \Gamma_2=\frac{\alpha-\beta}{\theta^2},\qquad
    \Gamma_3=\frac{\beta/2-3\Gamma_1}{\theta^2},\qquad \Gamma_4=\frac{1-\gamma}{\theta^2}
    \end{aligned}
    \tag{9, 40}
    $$
    """
    theta2 = ca.dot(xi, xi)
    use_series = theta2 < 1.0
    # 닫힌 해 쪽 분모, 급수 구간에서 0 나눗셈 방지
    th2 = ca.if_else(use_series, 1.0, theta2)
    th = ca.sqrt(th2)
    alpha_c = ca.sin(th) / th
    beta_c = 2.0 * (1.0 - ca.cos(th)) / th2
    gamma1_c = (1.0 - alpha_c) / th2
    gamma_c = 0.5 * th * ca.cos(0.5 * th) / ca.sin(0.5 * th)
    closed = {
        "alpha": alpha_c,
        "beta": beta_c,
        "gamma1": gamma1_c,
        "gamma2": (alpha_c - beta_c) / th2,
        "gamma3": (0.5 * beta_c - 3.0 * gamma1_c) / th2,
        "gamma4": (1.0 - gamma_c) / th2,
    }
    return {key: ca.if_else(use_series, _series(theta2, _COEFS[key]), closed[key]) for key in closed}


def _sym_prod(a, b):
    r"""
    $$
    \lceil a,b\rceil=\lceil a\rceil\lceil b\rceil+\lceil b\rceil\lceil a\rceil
    $$
    """
    hat_a, hat_b = ca.skew(a), ca.skew(b)
    return hat_a @ hat_b + hat_b @ hat_a


def dexp3(xi):
    r"""
    $$
    \mathrm{dexp}_{\xi}=I_3+\frac{\beta}{2}\lceil\xi\rceil+\Gamma_1\lceil\xi\rceil^2
    \tag{26}
    $$
    """
    c = _scalar_coefs(xi)
    hat_xi = ca.skew(xi)
    return ca.DM.eye(3) + 0.5 * c["beta"] * hat_xi + c["gamma1"] * hat_xi @ hat_xi


def dexp3_inv(xi):
    r"""
    $$
    \mathrm{dexp}_{\xi}^{-1}=I_3-\frac{1}{2}\lceil\xi\rceil+\Gamma_4\lceil\xi\rceil^2
    \tag{27}
    $$
    """
    c = _scalar_coefs(xi)
    hat_xi = ca.skew(xi)
    return ca.DM.eye(3) - 0.5 * hat_xi + c["gamma4"] * hat_xi @ hat_xi


def dexp6_block(xi, y):
    r"""
    $$
    C_{\xi}(y)=\frac{\beta}{2}\lceil y\rceil+\Gamma_1\lceil y,\xi\rceil+\Gamma_2\,\xi^{\top}y\lceil\xi\rceil
    +\Gamma_3\,\xi^{\top}y\lceil\xi\rceil^2
    \tag{30}
    $$
    """
    c = _scalar_coefs(xi)
    hat_xi = ca.skew(xi)
    xi_y = ca.dot(xi, y)
    return (
        0.5 * c["beta"] * ca.skew(y)
        + c["gamma1"] * _sym_prod(y, xi)
        + xi_y * (c["gamma2"] * hat_xi + c["gamma3"] * hat_xi @ hat_xi)
    )


def _zeta(xi, xi_dot, y):
    r"""
    $$
    \zeta(y)=\xi^{\top}y\,\frac{\xi^{\top}\dot{\xi}}{\lVert\xi\rVert^2},\qquad
    \lvert\zeta(y)\rvert\le\lVert y\rVert\lVert\dot{\xi}\rVert
    $$
    """
    theta2 = ca.dot(xi, xi)
    return ca.dot(xi, y) * ca.dot(xi, xi_dot) / ca.if_else(theta2 > 1e-30, theta2, 1.0)


def dexp6_block_dot(xi, xi_dot, y, y_dot):
    r"""
    $$
    \begin{aligned}
    \delta_0&=\dot{\xi}^{\top}y+\xi^{\top}\dot{y} \\
    \delta_1&=\xi^{\top}y\lceil\dot{\xi}\rceil+\xi^{\top}\dot{\xi}\lceil y\rceil+(\delta_0-4\zeta)\lceil\xi\rceil \\
    \delta_2&=\xi^{\top}y\lceil\xi,\dot{\xi}\rceil+\xi^{\top}\dot{\xi}\lceil\xi,y\rceil
    +(\delta_0-5\zeta)\lceil\xi\rceil^2 \\
    \dot{C}_{\xi,\dot{\xi}}(y,\dot{y})&=\frac{\beta}{2}\big(\lceil\dot{y}\rceil-\zeta\lceil\xi\rceil\big)
    +\Gamma_1\big(\lceil\dot{y},\xi\rceil+\lceil y,\dot{\xi}\rceil+\zeta\lceil\xi\rceil\big)
    +\Gamma_2\big(\delta_1+\zeta\lceil\xi\rceil^2\big)+\Gamma_3\,\delta_2
    \end{aligned}
    \tag{38, 41}
    $$

    $$
    \xi=0:\ \dot{C}_{0,\dot{\xi}}(y,\dot{y})=\frac{1}{2}\lceil\dot{y}\rceil+\frac{1}{6}\lceil\dot{\xi},y\rceil
    \tag{42}
    $$
    """
    c = _scalar_coefs(xi)
    zeta = _zeta(xi, xi_dot, y)
    hat_xi = ca.skew(xi)
    hat_xi2 = hat_xi @ hat_xi
    xi_y, xi_xid = ca.dot(xi, y), ca.dot(xi, xi_dot)
    delta0 = ca.dot(xi_dot, y) + ca.dot(xi, y_dot)
    delta1 = xi_y * ca.skew(xi_dot) + xi_xid * ca.skew(y) + (delta0 - 4.0 * zeta) * hat_xi
    delta2 = xi_y * _sym_prod(xi, xi_dot) + xi_xid * _sym_prod(xi, y) + (delta0 - 5.0 * zeta) * hat_xi2
    return (
        0.5 * c["beta"] * (ca.skew(y_dot) - zeta * hat_xi)
        + c["gamma1"] * (_sym_prod(y_dot, xi) + _sym_prod(y, xi_dot) + zeta * hat_xi)
        + c["gamma2"] * (delta1 + zeta * hat_xi2)
        + c["gamma3"] * delta2
    )


def dexp6(lam):
    r"""
    $$
    \mathrm{dexp}_{\lambda}=\begin{bmatrix}\mathrm{dexp}_{\xi}&C_{\xi}(\eta)\\ 0_3&\mathrm{dexp}_{\xi}\end{bmatrix}
    \tag{28}
    $$
    """
    eta, xi = lam[:3], lam[3:]
    diag = dexp3(xi)
    return ca.blockcat([[diag, dexp6_block(xi, eta)], [ca.DM.zeros(3, 3), diag]])


def dexp6_inv(lam):
    r"""
    $$
    \mathrm{dexp}_{\lambda}^{-1}=\begin{bmatrix}\mathrm{dexp}_{\xi}^{-1}&
    -\mathrm{dexp}_{\xi}^{-1}C_{\xi}(\eta)\,\mathrm{dexp}_{\xi}^{-1}\\ 0_3&\mathrm{dexp}_{\xi}^{-1}\end{bmatrix}
    \tag{29}
    $$
    """
    eta, xi = lam[:3], lam[3:]
    diag_inv = dexp3_inv(xi)
    return ca.blockcat([[diag_inv, -diag_inv @ dexp6_block(xi, eta) @ diag_inv], [ca.DM.zeros(3, 3), diag_inv]])


def dexp6_dot(lam, lam_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\lambda}=\begin{bmatrix}C_{\xi}(\dot{\xi})&\dot{C}_{\xi,\dot{\xi}}(\eta,\dot{\eta})\\
    0_3&C_{\xi}(\dot{\xi})\end{bmatrix}
    \tag{34, 36}
    $$
    """
    eta, xi = lam[:3], lam[3:]
    eta_dot, xi_dot = lam_dot[:3], lam_dot[3:]
    diag = dexp6_block(xi, xi_dot)
    return ca.blockcat([[diag, dexp6_block_dot(xi, xi_dot, eta, eta_dot)], [ca.DM.zeros(3, 3), diag]])
