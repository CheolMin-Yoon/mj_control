# Optimization 학습 로드맵

휴머노이드 SRBD MPC + WBC 컨텍스트에서 Boyd 『Convex Optimization』 책을 직접 구현하면서 학습. 각 항목은 **현재 코드(`mj_sim/humanoid`)와 어떻게 연결되는지** 명시.

---

## 우선순위 표

★★★ 즉시 실용 / ★★ 중요 / ★ 부차적 / — 학습용

---

## Part I — Theory

### Ch 4. Convex optimization problems

| 절 | 구현 항목 | 우선 | 메모 |
|---|---|---|---|
| 4.4 Quadratic optimization | KKT 행렬로 등식제약 QP 직접 풀이 | ★★ | WBC/MPC가 QP. KKT linear system이 가장 간단한 풀이 형태 |
| 4.6 Generalized inequality constraints | **Friction cone SOCP** | ★★ | 현재 WBC는 5면 pyramid 보수 근사. 정확한 quadratic cone (`||f_xy|| ≤ μ·f_z`)를 CVXPY+ECOS로 5줄. stance 영역 ~30% 확장 가능 |

### Ch 5. Duality

| 절 | 구현 항목 | 우선 | 메모 |
|---|---|---|---|
| 5.5 Optimality conditions (KKT) | **KKT 조건 직접 검증** | ★★ | ProxQP 솔루션이 정말 KKT 만족하는지 체크. WBC 발산 시 디버깅 무기 |
| 5.6 Perturbation/sensitivity | dual 변수로 active constraint 식별 | ★ | WBC에서 어느 마찰원뿔이 binding인지 자동 감지 |

---

## Part II — Applications

### Ch 6. Approximation and fitting

| 절 | 구현 항목 | 우선 | 메모 |
|---|---|---|---|
| 6.3 Regularized approximation | ℓ₁ regularized LS (LASSO) | ★ | gait 게인/접촉 force sparse 식별 |
| 6.4 Robust approximation | M-estimator robust LS | ★ | force/torque 센서 노이즈 fitting (실기 옮길 때 가치) |

### Ch 7. Statistical estimation

거의 무관. 패스.

### Ch 8. Geometric problems

| 절 | 구현 항목 | 우선 | 메모 |
|---|---|---|---|
| 8.4 Extremal volume ellipsoids | Löwner-John 외/내접 ellipsoid | ★★ | capture region, reachable set 시각화. ALIP 관성과 결합하면 "발 안 옮기고 살아남는 base velocity 영역" |
| 8.5 Centering | **Support polygon analytic/Chebyshev center** | ★★★ | 휴머노이드 stability margin. 현재 `g1_main.ipynb`에 ConvexHull만 있고 안쪽 중심 미계산. **반나절 작업** |

---

## Part III — Algorithms

### Ch 9. Unconstrained minimization

| 절 | 구현 항목 | 우선 | 메모 |
|---|---|---|---|
| 9.3 Gradient descent | (학습용 워밍업) | — | Newton에 가려짐 |
| 9.5 Newton's method | **Newton + backtracking line search** | ★★★ | IK Levenberg-Marquardt, DDP/iLQR, IK 트래킹의 building block |
| 9.7 Implementation | sparse Hessian factorize | ★ | 구조 활용 |

### Ch 10. Equality constrained minimization

| 절 | 구현 항목 | 우선 | 메모 |
|---|---|---|---|
| 10.2 Newton with eq constraints | KKT system Newton | ★★ | dynamics consistency 같은 등식 제약 QP 직접. SRBD MPC 핵심 |
| 10.3 Infeasible start Newton | infeasible-start IPM 시작점 | ★ | Ch 11과 연결 |

### Ch 11. Interior-point methods

| 절 | 구현 항목 | 우선 | 메모 |
|---|---|---|---|
| 11.2 Log barrier + central path | 방벽 함수 시각화 | ★ | central path 추적 그림. 직관 형성용 |
| 11.3 The barrier method | Barrier IPM 풀체인 | ★★ | ProxQP 내부 이해. **3주 투자** |
| 11.7 Primal-dual IPM | primal-dual variant | ★ | 11.3 마치고 |

---

## 별책 — Boyd ADMM Monograph (책 외, 가장 실용)

『Distributed Optimization and Statistical Learning via ADMM』 (2011, 무료 PDF)

| 항목 | 우선 | 메모 |
|---|---|---|
| **ADMM-for-QP from scratch** | ★★★ | 50줄 python. OSQP, Pinocchio 4.0 `ADMMConstraintSolver`의 본체. IPM(★★, 3주)보다 적은 비용으로 가시적 결과 |
| Adaptive ρ scheduling | ★★ | 수렴 가속. OSQP 핵심 트릭 |
| Warm-start MPC | ★★ | MPC 매 step 5~10 iter 수렴 |

핵심 식:
```
x ← (P + ρI)⁻¹ (ρz - y - q)    # KKT 풀이 (Cholesky 사전인수분해)
z ← Π_C(x + y/ρ)                # constraint set으로 projection
y ← y + ρ(x - z)                # dual update
```

`P + ρI` Cholesky 한 번 하고 매 iter forward/back substitution + projection만 → 학부생도 이해 가능.

---

## 추천 학습 루트 (목차 순 + 실용성 가중)

```
Ch 4.4 (QP 풀이)
  └→ Ch 5 (Duality, KKT)
       ├→ [별책] ADMM-for-QP 50줄                    ★★★
       └→ Ch 4.6 (Friction cone SOCP)
            └→ Ch 8.5 (Support polygon center)        ★★★
                 └→ Ch 8.4 (Ellipsoid capture region)
                      └→ Ch 9.5 (Newton + line search) ★★★
                           └→ Ch 10.2 (eq-constrained Newton)
                                └→ Ch 11.3 (Barrier IPM)
```

★★★ 4개(8.5 / 9.5 / 10.2 / ADMM)만 완료해도 휴머노이드 SRBD MPC + WBC를 자기 구현 솔버로 풀 수 있는 수준 도달.

---

## 참고 자료

- **Boyd & Vandenberghe**, *Convex Optimization* — 본 학습 대상. 무료 PDF: https://web.stanford.edu/~boyd/cvxbook/
- **Boyd et al.**, *Distributed Optimization via ADMM* monograph — 무료 PDF: https://web.stanford.edu/~boyd/papers/admm_distr_stats.html
- **Stellato et al.**, *OSQP: An Operator Splitting Solver for Quadratic Programs* (2020) — 사실상 ADMM-for-QP 논문
- **Wieber, Tedrake, Posa**, locomotion 관련 convex opt 응용 자료

---

## 디렉토리 구성 (예정)

```
optimization/
├── README.md                    # 이 파일
├── admm.py                      # ADMM-for-QP from scratch
├── support_polygon_center.py    # Ch 8.5 stability margin
├── newton_line_search.py        # Ch 9.5
├── barrier_ipm.py               # Ch 11.3 (long-term)
└── pinocchio_examples/          # pin 4.0 ADMM 참고 예제 (외부 코드)
    ├── admm-constraint-solver.py
    └── g1-constraint-simulation.py
```
