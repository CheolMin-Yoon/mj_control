1. Basic Derivation + Local stabilization of nonlinear systems

카트폴을 upright 평형점에서 선형화 → A, B 행렬 도출
무한구간 LQR로 K 게인 구해서 비선형 시스템에 적용
Region of attraction 실험 (초기 각도 얼마까지 안정화되나?)
→ LQR의 출발점. 무조건 해야 함


2. Finite-horizon LQR + Time-varying LQR
Riccati ODE를 backward로 적분해서 P(t), K(t) 얻기
무한구간 게인과 비교 (terminal cost 영향)
→ 이론과 구현의 차이를 체감하기 좋음

3. Local trajectory stabilization (TVLQR)
Swing-up 궤적(예: direct collocation으로 미리 계산)을 따라가는 TVLQR
비선형 시스템에서 nominal trajectory 주변 선형화 → 시간가변 A(t), B(t)
→ 카트폴의 꽃. swing-up + balance를 LQR만으로 푸는 경험

4. Discrete-time Riccati Equations
시뮬레이션은 어차피 이산이라 DARE가 더 실용적
continuous vs discrete 게인 비교
→ 실제 구현에 직결