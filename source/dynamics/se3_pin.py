r"""
$$
\begin{aligned}
V&=\begin{bmatrix}v\\ \omega\end{bmatrix},\qquad \lambda=\begin{bmatrix}\eta\\ \xi\end{bmatrix} \\
\texttt{pin.Jexp3}(\xi)&=\mathrm{dexp}_{-\xi}=\mathrm{dexp}_{\xi}^{\top} \\
\texttt{pin.Jexp6}(\lambda)&=\mathrm{dexp}_{-\lambda},\qquad
\mathrm{dexp}_{\lambda}=\mathrm{Ad}_{\exp(\lambda)}\,\mathrm{dexp}_{-\lambda}
\end{aligned}
$$
"""

import numpy as np
import pinocchio as pin


def hat3(w):
    r"""
    $$
    \lceil\omega\rceil=\texttt{pin.skew}(\omega)
    \tag{1}
    $$
    """
    return pin.skew(w)


def vee3(W):
    r"""
    $$
    \lfloor W\rfloor=\texttt{pin.unSkew}(W)
    $$
    """
    return pin.unSkew(W)


def hat6(V):
    r"""
    $$
    \lceil V\rceil=\begin{bmatrix}\lceil\omega\rceil&v\\ 0_{1\times3}&0\end{bmatrix}
    =\texttt{Motion}(V)\texttt{.homogeneous}
    \tag{2}
    $$
    """
    return pin.Motion(V).homogeneous


def vee6(M):
    r"""
    $$
    \lfloor M\rfloor=\begin{bmatrix}M_{1:3,4}\\ \lfloor M_{1:3,1:3}\rfloor\end{bmatrix}
    $$
    """
    return pin.Motion(M[:3, 3], pin.unSkew(M[:3, :3])).vector


def Ad(T):  # noqa: N802
    r"""
    $$
    \mathrm{Ad}_T=\begin{bmatrix}R&\lceil r\rceil R\\ 0_3&R\end{bmatrix}=\texttt{SE3}(T)\texttt{.action}
    \tag{5}
    $$
    """
    return pin.SE3(T).action


def ad(V):
    r"""
    $$
    \mathrm{ad}_V=\begin{bmatrix}\lceil\omega\rceil&\lceil v\rceil\\ 0_3&\lceil\omega\rceil\end{bmatrix}
    =\texttt{Motion}(V)\texttt{.action}
    \tag{3}
    $$
    """
    return pin.Motion(V).action


def exp3(xi):
    r"""
    $$
    \exp(\xi)=I_3+\alpha\lceil\xi\rceil+\frac{\beta}{2}\lceil\xi\rceil^2
    \tag{8}
    $$
    """
    return pin.exp3(xi)


def log3(R):
    r"""
    $$
    \xi=\log(R),\qquad \lVert\xi\rVert<\pi
    \tag{11}
    $$
    """
    return pin.log3(R)


def dexp3(xi):
    r"""
    $$
    \mathrm{dexp}_{\xi}=\texttt{Jexp3}(\xi)^{\top}
    \tag{26}
    $$
    """
    return pin.Jexp3(xi).T


def dexp3_inv(xi):
    r"""
    $$
    \begin{aligned}
    \mathrm{dexp}_{\xi}^{-1}&=I_3-\frac{1}{2}\lceil\xi\rceil+\Gamma_4\lceil\xi\rceil^2 \\
    \Gamma_4&=\frac{1-\gamma}{\lVert\xi\rVert^2},\qquad \gamma=\frac{\lVert\xi\rVert}{2}\cot\frac{\lVert\xi\rVert}{2}
    \approx\frac{1}{12}+\frac{\lVert\xi\rVert^2}{720}+\frac{\lVert\xi\rVert^4}{30240}
    \end{aligned}
    \tag{27}
    $$

    $$
    \lVert\xi\rVert<10^{-2}:\ \text{Taylor}
    $$
    """
    theta2 = float(xi @ xi)
    if theta2 < 1e-4:
        gamma4 = 1.0 / 12.0 + theta2 / 720.0 + theta2**2 / 30240.0
    else:
        half = 0.5 * np.sqrt(theta2)
        gamma4 = (1.0 - half / np.tan(half)) / theta2
    hat_xi = pin.skew(xi)
    return np.eye(3) - 0.5 * hat_xi + gamma4 * hat_xi @ hat_xi


def dexp3_dot(xi, xi_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\xi}=C_{\xi}(\dot{\xi})
    \tag{34}
    $$
    """
    return dexp6_block(xi, xi_dot)


def dexp3_inv_dot(xi, xi_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\xi}^{-1}=D_{\xi}(\dot{\xi})
    =-\mathrm{dexp}_{\xi}^{-1}\Big(\frac{d}{dt}\mathrm{dexp}_{\xi}\Big)\mathrm{dexp}_{\xi}^{-1}
    \tag{35}
    $$
    """
    dexp_inv = dexp3_inv(xi)
    return -dexp_inv @ dexp3_dot(xi, xi_dot) @ dexp_inv


def exp6(lam):
    r"""
    $$
    \exp(\lambda)=\begin{bmatrix}\exp(\xi)&\mathrm{dexp}_{\xi}\,\eta\\ 0_{1\times3}&1\end{bmatrix}
    \tag{10}
    $$
    """
    return pin.exp6(lam).homogeneous


def log6(T):
    r"""
    $$
    \log(T)=\begin{bmatrix}\eta\\ \xi\end{bmatrix},\qquad \eta=\mathrm{dexp}_{\xi}^{-1}r
    \tag{12}
    $$
    """
    return pin.log6(pin.SE3(T)).vector


def dexp6(lam):
    r"""
    $$
    \mathrm{dexp}_{\lambda}=\begin{bmatrix}\mathrm{dexp}_{\xi}&C_{\xi}(\eta)\\ 0_3&\mathrm{dexp}_{\xi}\end{bmatrix}
    \tag{28}
    $$
    """
    eta, xi = lam[:3], lam[3:]
    diag = dexp3(xi)
    return np.block([[diag, dexp6_block(xi, eta)], [np.zeros((3, 3)), diag]])


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
    return np.block([[diag_inv, -diag_inv @ dexp6_block(xi, eta) @ diag_inv], [np.zeros((3, 3)), diag_inv]])


def dexp6_dot(lam, lam_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\lambda}=\begin{bmatrix}C_{\xi}(\dot{\xi})&\dot{C}_{\xi,\dot{\xi}}(\eta,\dot{\eta})\\
    0_3&C_{\xi}(\dot{\xi})\end{bmatrix}
    \tag{36}
    $$
    """
    eta, xi = lam[:3], lam[3:]
    eta_dot, xi_dot = lam_dot[:3], lam_dot[3:]
    diag = dexp3_dot(xi, xi_dot)
    return np.block([[diag, dexp6_block_dot(xi, xi_dot, eta, eta_dot)], [np.zeros((3, 3)), diag]])


def dexp6_inv_dot(lam, lam_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\lambda}^{-1}=-\mathrm{dexp}_{\lambda}^{-1}\Big(\frac{d}{dt}\mathrm{dexp}_{\lambda}\Big)
    \mathrm{dexp}_{\lambda}^{-1}
    \tag{37}
    $$
    """
    dexp_inv = dexp6_inv(lam)
    return -dexp_inv @ dexp6_dot(lam, lam_dot) @ dexp_inv


def _half_angle_coefs(xi):
    r"""
    $$
    \begin{aligned}
    s&=\frac{\sin(\lVert\xi\rVert/2)}{\lVert\xi\rVert/2},\qquad c=\cos(\lVert\xi\rVert/2) \\
    \alpha&=sc=\frac{\sin\lVert\xi\rVert}{\lVert\xi\rVert} \\
    \beta&=s^2=\frac{2(1-\cos\lVert\xi\rVert)}{\lVert\xi\rVert^2}
    \end{aligned}
    \tag{9}
    $$
    """
    theta = np.linalg.norm(xi)
    s = np.sinc(theta / (2.0 * np.pi))
    return s * np.cos(theta / 2.0), s * s


def _sym_prod(a, b):
    r"""
    $$
    \lceil a,b\rceil=\lceil a\rceil\lceil b\rceil+\lceil b\rceil\lceil a\rceil
    $$
    """
    hat_a, hat_b = pin.skew(a), pin.skew(b)
    return hat_a @ hat_b + hat_b @ hat_a


def _gammas(xi):
    r"""
    $$
    \begin{aligned}
    \Gamma_1&=\frac{1-\alpha}{\lVert\xi\rVert^2}\approx\frac{1}{6}-\frac{\lVert\xi\rVert^2}{120}
    +\frac{\lVert\xi\rVert^4}{5040} \\
    \Gamma_2&=\frac{\alpha-\beta}{\lVert\xi\rVert^2}\approx-\frac{1}{12}+\frac{\lVert\xi\rVert^2}{180}
    -\frac{\lVert\xi\rVert^4}{6720} \\
    \Gamma_3&=\frac{\beta/2-3\Gamma_1}{\lVert\xi\rVert^2}\approx-\frac{1}{60}+\frac{\lVert\xi\rVert^2}{1260}
    \end{aligned}
    \tag{40}
    $$

    $$
    \lVert\xi\rVert<10^{-2}:\ \text{Taylor}
    $$
    """
    theta2 = float(xi @ xi)
    if theta2 < 1e-4:
        gamma1 = 1.0 / 6.0 - theta2 / 120.0 + theta2**2 / 5040.0
        gamma2 = -1.0 / 12.0 + theta2 / 180.0 - theta2**2 / 6720.0
        gamma3 = -1.0 / 60.0 + theta2 / 1260.0
        return gamma1, gamma2, gamma3
    alpha, beta = _half_angle_coefs(xi)
    gamma1 = (1.0 - alpha) / theta2
    gamma2 = (alpha - beta) / theta2
    gamma3 = (beta / 2.0 - 3.0 * gamma1) / theta2
    return gamma1, gamma2, gamma3


def dexp6_block(xi, y):
    r"""
    $$
    \begin{aligned}
    C_{\xi}(y)={}&\frac{\beta}{2}\lceil y\rceil+\frac{1-\alpha}{\lVert\xi\rVert^2}\lceil y,\xi\rceil
    +\frac{\alpha-\beta}{\lVert\xi\rVert^2}\xi^{\top}y\lceil\xi\rceil
    -\frac{1}{\lVert\xi\rVert^2}\Big(\frac{3(1-\alpha)}{\lVert\xi\rVert^2}-\frac{\beta}{2}\Big)
    \xi^{\top}y\lceil\xi\rceil^2 \\
    ={}&\frac{\beta}{2}\lceil y\rceil+\Gamma_1\lceil y,\xi\rceil+\Gamma_2\,\xi^{\top}y\lceil\xi\rceil
    +\Gamma_3\,\xi^{\top}y\lceil\xi\rceil^2
    \end{aligned}
    \tag{30}
    $$
    """
    _, beta = _half_angle_coefs(xi)
    gamma1, gamma2, gamma3 = _gammas(xi)
    hat_xi = pin.skew(xi)
    xi_y = xi @ y
    return 0.5 * beta * pin.skew(y) + gamma1 * _sym_prod(y, xi) + xi_y * (gamma2 * hat_xi + gamma3 * hat_xi @ hat_xi)


def _zeta(xi, xi_dot, y):
    r"""
    $$
    \zeta(y)=\xi^{\top}y\,\frac{\xi^{\top}\dot{\xi}}{\lVert\xi\rVert^2}
    $$
    """
    return (xi @ y) * (xi @ xi_dot) / (xi @ xi)


def _deltas(xi, xi_dot, y, y_dot):
    r"""
    $$
    \begin{aligned}
    \delta_0(y,\dot{y})&=\dot{\xi}^{\top}y+\xi^{\top}\dot{y} \\
    \delta_1(y,\dot{y})&=\xi^{\top}y\lceil\dot{\xi}\rceil+\xi^{\top}\dot{\xi}\lceil y\rceil
    +\big(\delta_0(y,\dot{y})-4\zeta(y)\big)\lceil\xi\rceil \\
    \delta_2(y,\dot{y})&=\xi^{\top}y\lceil\xi,\dot{\xi}\rceil+\xi^{\top}\dot{\xi}\lceil\xi,y\rceil
    +\big(\delta_0(y,\dot{y})-5\zeta(y)\big)\lceil\xi\rceil^2
    \end{aligned}
    \tag{41}
    $$
    """
    zeta = _zeta(xi, xi_dot, y)
    delta0 = xi_dot @ y + xi @ y_dot
    hat_xi = pin.skew(xi)
    delta1 = (xi @ y) * pin.skew(xi_dot) + (xi @ xi_dot) * pin.skew(y) + (delta0 - 4.0 * zeta) * hat_xi
    delta2 = (
        (xi @ y) * _sym_prod(xi, xi_dot) + (xi @ xi_dot) * _sym_prod(xi, y) + (delta0 - 5.0 * zeta) * hat_xi @ hat_xi
    )
    return delta1, delta2


def dexp6_block_dot(xi, xi_dot, y, y_dot):
    r"""
    $$
    \begin{aligned}
    \dot{C}_{\xi,\dot{\xi}}(y,\dot{y})={}&\frac{\beta}{2}\big(\lceil\dot{y}\rceil-\zeta(y)\lceil\xi\rceil\big)
    +\Gamma_1\big(\lceil\dot{y},\xi\rceil+\lceil y,\dot{\xi}\rceil+\zeta(y)\lceil\xi\rceil\big) \\
    &+\Gamma_2\big(\delta_1(y,\dot{y})+\zeta(y)\lceil\xi\rceil^2\big)+\Gamma_3\,\delta_2(y,\dot{y})
    \end{aligned}
    \tag{38}
    $$

    $$
    \dot{C}_{0,\dot{\xi}}(y,\dot{y})=\frac{1}{2}\lceil\dot{y}\rceil+\frac{1}{6}\lceil\dot{\xi},y\rceil
    \tag{42}
    $$
    """
    if xi @ xi < 1e-24:
        return 0.5 * pin.skew(y_dot) + _sym_prod(xi_dot, y) / 6.0
    _, beta = _half_angle_coefs(xi)
    gamma1, gamma2, gamma3 = _gammas(xi)
    zeta = _zeta(xi, xi_dot, y)
    delta1, delta2 = _deltas(xi, xi_dot, y, y_dot)
    hat_xi = pin.skew(xi)
    return (
        0.5 * beta * (pin.skew(y_dot) - zeta * hat_xi)
        + gamma1 * (_sym_prod(y_dot, xi) + _sym_prod(y, xi_dot) + zeta * hat_xi)
        + gamma2 * (delta1 + zeta * hat_xi @ hat_xi)
        + gamma3 * delta2
    )
