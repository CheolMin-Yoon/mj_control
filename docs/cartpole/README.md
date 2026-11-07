# Cartpole

상태 4차원(`[cart_pos, pole_angle, cart_vel, pole_vel]`), 입력 1차원의 최적제어 기초. MuJoCo 장면은
`source/robots/cartpole/xmls/scene.xml`이고 actuator는 `motor`(gear 50)다 → 선형 모델의 B에 gear를 곱한다. Physics timestep이
0.01 s라 제어도 100 Hz다.

## 구현

| 위치 | 내용 |
| --- | --- |
| `scripts/cartpole_lqr.py` | upright 선형화 + LQR, pole 0.1 rad 교란에서 시작, 상태는 `MujocoKernel.qpos`·`qvel` 전체 |
| `source/dynamics/cartpole.py` | `CartpoleDynamics`: 질점 pole 해석 동역학 `continuous_dynamics`, 평형점 `linearize` |
| `source/control/lqr.py` | `lqr_gain`: `solve_continuous_are` 연속시간 무한구간 LQR |
| `source/dynamics/discretization.py` | 비선형 `euler`·`rk4`, 선형 `zoh`(expm)·`taylor1`·`taylor2`, DARE·이산 MPC용 |

## 학습 목록

LQR
1. 평형점 선형화 → 무한구간 LQR → region of attraction(초기 각도 한계)
2. Finite-horizon LQR: Riccati ODE backward 적분으로 P(t)·K(t), 무한구간 게인과 비교
3. TVLQR: swing-up 궤적(direct collocation 등) 주변 시간가변 A(t)·B(t)로 추종
4. DARE: 이산 Riccati, 연속·이산 게인 비교

MPC
1. Linear MPC: 선형화 후 QP(OSQP)
2. Condensed 대 sparse formulation의 QP 차원과 속도
3. 제약(카트 ±1 m, 힘 ±10 N)이 게인에 미치는 영향
4. Horizon N = 5·20·50 비교, terminal cost = LQR cost의 안정성
5. NMPC와 선형 MPC의 swing-up 차이, saturation 영역에서 MPC 대 LQR
