r"""
$$
\begin{aligned}
V&=\begin{bmatrix}v\\ \omega\end{bmatrix},\qquad \lambda=\begin{bmatrix}\eta\\ \xi\end{bmatrix} \\
\texttt{pin.Jexp6}(\lambda)&=\mathrm{dexp}_{-\lambda} \\
\mathrm{dexp}_{\lambda}&=\mathrm{Ad}_{\exp(\lambda)}\,\mathrm{dexp}_{-\lambda}
\end{aligned}
$$
"""

import numpy as np


def skew(w):
    r"""
    $$
    \lceil\omega\rceil=\begin{bmatrix}0&-\omega_3&\omega_2\\ \omega_3&0&-\omega_1\\ -\omega_2&\omega_1&0\end{bmatrix}
    \tag{1}
    $$
    """
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ]
    )


def unskew(W):
    r"""
    $$
    \begin{aligned}
    \lfloor W\rfloor&=\begin{bmatrix}W_{32}\\ W_{13}\\ W_{21}\end{bmatrix} \\
    \lfloor\lceil\omega\rceil\rfloor&=\omega
    \end{aligned}
    $$
    """
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def hat6(V):
    r"""
    $$
    \lceil V\rceil=\begin{bmatrix}\lceil\omega\rceil&v\\ 0_{1\times3}&0\end{bmatrix}
    \tag{2}
    $$
    """
    raise NotImplementedError


def vee6(M):
    r"""
    $$
    \begin{aligned}
    \lfloor M\rfloor&=\begin{bmatrix}M_{1:3,4}\\ \lfloor M_{1:3,1:3}\rfloor\end{bmatrix} \\
    \lfloor\lceil V\rceil\rfloor&=V
    \end{aligned}
    $$
    """
    raise NotImplementedError


def adjoint(T):
    r"""
    $$
    \begin{aligned}
    T&=\begin{bmatrix}R&r\\ 0_{1\times3}&1\end{bmatrix} \\
    \mathrm{Ad}_T&=\begin{bmatrix}R&\lceil r\rceil R\\ 0_3&R\end{bmatrix}
    \end{aligned}
    \tag{5}
    $$
    """
    raise NotImplementedError


def ad(V):
    r"""
    $$
    \mathrm{ad}_V=\begin{bmatrix}\lceil\omega\rceil&\lceil v\rceil\\ 0_3&\lceil\omega\rceil\end{bmatrix}
    \tag{3}
    $$
    """
    raise NotImplementedError


def _half_angle_coefs(xi):
    r"""
    $$
    \begin{aligned}
    s&=\frac{\sin(\lVert\xi\rVert/2)}{\lVert\xi\rVert/2} \\
    c&=\cos(\lVert\xi\rVert/2) \\
    \alpha&=sc \\
    \beta&=s^2 \\
    \gamma&=\frac{\alpha}{\beta}
    \end{aligned}
    \tag{9}
    $$
    """
    raise NotImplementedError


def exp3(xi):
    r"""
    $$
    \exp(\xi)=I_3+\alpha\lceil\xi\rceil+\frac{\beta}{2}\lceil\xi\rceil^2
    \tag{8}
    $$
    """
    raise NotImplementedError


def log3(R):
    r"""
    $$
    \begin{aligned}
    \xi&=\log(R) \\
    \lVert\xi\rVert&<\pi
    \end{aligned}
    \tag{11}
    $$
    """
    raise NotImplementedError


def dexp3(xi):
    r"""
    $$
    \begin{aligned}
    \mathrm{dexp}_{\xi}&=I_3+\frac{\beta}{2}\lceil\xi\rceil+\frac{1-\alpha}{\lVert\xi\rVert^2}\lceil\xi\rceil^2 \\
    \mathrm{dexp}_{0}&=I_3
    \end{aligned}
    \tag{26}
    $$
    """
    raise NotImplementedError


def dexp3_inv(xi):
    r"""
    $$
    \begin{aligned}
    \mathrm{dexp}_{\xi}^{-1}&=I_3-\frac{1}{2}\lceil\xi\rceil+\frac{1-\gamma}{\lVert\xi\rVert^2}\lceil\xi\rceil^2 \\
    \mathrm{dexp}_{0}^{-1}&=I_3 \\
    \lVert\xi\rVert&<2\pi
    \end{aligned}
    \tag{27}
    $$
    """
    raise NotImplementedError


def _sym_prod(a, b):
    r"""
    $$
    \lceil a,b\rceil=\lceil a\rceil\lceil b\rceil+\lceil b\rceil\lceil a\rceil
    $$
    """
    raise NotImplementedError


def C(xi, eta):  # noqa: N802
    r"""
    $$
    \begin{aligned}
    C_{\xi}(\eta)={}&\frac{\beta}{2}\lceil\eta\rceil+\frac{1-\alpha}{\lVert\xi\rVert^2}\lceil\eta,\xi\rceil
    +\frac{\alpha-\beta}{\lVert\xi\rVert^2}\xi^{\top}\eta\lceil\xi\rceil \\
    &-\frac{1}{\lVert\xi\rVert^2}\Big(\frac{3(1-\alpha)}{\lVert\xi\rVert^2}-\frac{\beta}{2}\Big)
    \xi^{\top}\eta\lceil\xi\rceil^2 \\
    C_{0}(\eta)={}&\frac{1}{2}\lceil\eta\rceil
    \end{aligned}
    \tag{30}
    $$
    """
    raise NotImplementedError


def D(xi, eta):  # noqa: N802
    r"""
    $$
    \begin{aligned}
    D_{\xi}(\eta)={}&-\frac{1}{2}\lceil\eta\rceil+\frac{1-\gamma}{\lVert\xi\rVert^2}\lceil\eta,\xi\rceil
    +\frac{1}{\lVert\xi\rVert^2}\frac{1/\beta+\gamma-2}{\lVert\xi\rVert^2}\xi^{\top}\eta\lceil\xi\rceil^2 \\
    D_{0}(\eta)={}&-\frac{1}{2}\lceil\eta\rceil
    \end{aligned}
    \tag{31}
    $$
    """
    raise NotImplementedError


def exp6(lam):
    r"""
    $$
    \exp(\lambda)=\begin{bmatrix}\exp(\xi)&\mathrm{dexp}_{\xi}\,\eta\\ 0_{1\times3}&1\end{bmatrix}
    \tag{10}
    $$
    """
    raise NotImplementedError


def log6(T):
    r"""
    $$
    \begin{aligned}
    \log(T)&=\begin{bmatrix}\eta\\ \xi\end{bmatrix} \\
    \xi&=\log(R) \\
    \eta&=\mathrm{dexp}_{\xi}^{-1}r
    \end{aligned}
    \tag{12}
    $$
    """
    raise NotImplementedError


def dexp6(lam):
    r"""
    $$
    \begin{aligned}
    \mathrm{dexp}_{\lambda}&=\begin{bmatrix}\mathrm{dexp}_{\xi}&C_{\xi}(\eta)\\ 0_3&\mathrm{dexp}_{\xi}\end{bmatrix} \\
    V_s&=\mathrm{dexp}_{\lambda}\dot{\lambda}
    \end{aligned}
    \tag{28}
    $$
    """
    raise NotImplementedError


def dexp6_inv(lam):
    r"""
    $$
    \begin{aligned}
    \mathrm{dexp}_{\lambda}^{-1}&=\begin{bmatrix}\mathrm{dexp}_{\xi}^{-1}&D_{\xi}(\eta)\\
    0_3&\mathrm{dexp}_{\xi}^{-1}\end{bmatrix} \\
    \dot{\lambda}&=\mathrm{dexp}_{\lambda}^{-1}V_s
    \end{aligned}
    \tag{29}
    $$
    """
    raise NotImplementedError


def _gammas(xi):
    r"""
    $$
    \begin{aligned}
    \Gamma_1&=\frac{1-\alpha}{\lVert\xi\rVert^2} \\
    \Gamma_2&=\frac{\alpha-\beta}{\lVert\xi\rVert^2} \\
    \Gamma_3&=\frac{1}{\lVert\xi\rVert^2}\Big(\frac{\beta}{2}-\frac{3(1-\alpha)}{\lVert\xi\rVert^2}\Big)
    =\frac{\beta/2-3\Gamma_1}{\lVert\xi\rVert^2} \\
    \Gamma_4&=\frac{1-\gamma}{\lVert\xi\rVert^2}=-\frac{\Gamma_2}{\beta} \\
    \Gamma_5&=\frac{1}{\lVert\xi\rVert^2}\frac{1/\beta+\gamma-2}{\lVert\xi\rVert^2}
    =\frac{\Gamma_1+2\Gamma_2}{\lVert\xi\rVert^2\beta}
    \end{aligned}
    \tag{40}
    $$
    """
    raise NotImplementedError


def _zeta(xi, xi_dot, y):
    r"""
    $$
    \zeta(y)=\xi^{\top}y\,\frac{\xi^{\top}\dot{\xi}}{\lVert\xi\rVert^2}
    $$
    """
    raise NotImplementedError


def _deltas(xi, xi_dot, y, y_dot):
    r"""
    $$
    \begin{aligned}
    \delta_0(y,\dot{y})&=\dot{\xi}^{\top}y+\xi^{\top}\dot{y} \\
    \delta_1(y,\dot{y})&=\xi^{\top}y\lceil\dot{\xi}\rceil+\xi^{\top}\dot{\xi}\lceil y\rceil
    +\big(\delta_0(y,\dot{y})-4\zeta(y)\big)\lceil\xi\rceil \\
    \delta_2(y,\dot{y})&=\xi^{\top}y\lceil\xi,\dot{\xi}\rceil+\xi^{\top}\dot{\xi}\lceil\xi,y\rceil
    +\big(\delta_0(y,\dot{y})-5\zeta(y)\big)\lceil\xi\rceil^2 \\
    \delta_3(y,\dot{y})&=\xi^{\top}y\lceil\xi,\dot{\xi}\rceil+\xi^{\top}\dot{\xi}\lceil\xi,y\rceil
    +\big(\delta_0(y,\dot{y})-3\zeta(y)\big)\lceil\xi\rceil^2
    \end{aligned}
    \tag{41}
    $$
    """
    raise NotImplementedError


def C_dot(xi, xi_dot, y, y_dot):  # noqa: N802
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
    raise NotImplementedError


def D_dot(xi, xi_dot, y, y_dot):  # noqa: N802
    r"""
    $$
    \begin{aligned}
    \dot{D}_{\xi,\dot{\xi}}(y,\dot{y})={}&-\frac{1}{2}\lceil\dot{y}\rceil
    +\frac{2}{\lVert\xi\rVert^2}\Big(1-\frac{\gamma}{\beta}\Big)\zeta(y)\lceil\xi\rceil^2 \\
    &+\Gamma_4\big(\lceil\dot{y},\xi\rceil+\lceil y,\dot{\xi}\rceil\big)+\Gamma_5\,\delta_3(y,\dot{y})
    \end{aligned}
    \tag{39}
    $$

    $$
    \dot{D}_{0,\dot{\xi}}(y,\dot{y})=-\frac{1}{2}\lceil\dot{y}\rceil+\frac{1}{12}\lceil\dot{\xi},y\rceil
    \tag{43}
    $$
    """
    raise NotImplementedError


def dexp3_dot(xi, xi_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\xi}=C_{\xi}(\dot{\xi})
    \tag{34}
    $$
    """
    raise NotImplementedError


def dexp3_inv_dot(xi, xi_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\xi}^{-1}=D_{\xi}(\dot{\xi})
    \tag{35}
    $$
    """
    raise NotImplementedError


def dexp6_dot(lam, lam_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\lambda}=\begin{bmatrix}C_{\xi}(\dot{\xi})&\dot{C}_{\xi,\dot{\xi}}(\eta,\dot{\eta})\\
    0_3&C_{\xi}(\dot{\xi})\end{bmatrix}
    \tag{36}
    $$
    """
    raise NotImplementedError


def dexp6_inv_dot(lam, lam_dot):
    r"""
    $$
    \frac{d}{dt}\mathrm{dexp}_{\lambda}^{-1}=\begin{bmatrix}D_{\xi}(\dot{\xi})&\dot{D}_{\xi,\dot{\xi}}(\eta,\dot{\eta})\\
    0_3&D_{\xi}(\dot{\xi})\end{bmatrix}
    \tag{37}
    $$
    """
    raise NotImplementedError
