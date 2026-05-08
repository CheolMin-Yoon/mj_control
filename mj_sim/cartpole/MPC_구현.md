카트폴: MPC "기초 체화용"
상태 4차원, 입력 1차원 → 손으로 디버깅 가능. 여기서 다음을 모두 검증하세요:

Linear MPC: 선형화 후 QP 풀기 (OSQP/quadprog)
Condensed vs Sparse formulation: QP 차원과 속도 비교
Constraint 효과: 카트 위치 ±1m, 힘 ±10N 같은 제약이 게인을 어떻게 바꾸는지
Horizon length 영향: N=5 vs N=20 vs N=50
Terminal cost = LQR cost: stability 보장 (실험적으로 확인)
Nonlinear MPC (NMPC): 선형 MPC와 swing-up 성능 차이
MPC vs LQR: saturation 발생 영역에서 비교
→ 이론 검증과 솔버 감각 잡는 단계. 모든 변형을 다 해보세요.


Go2: MPC "실전용"
여기서는 카트폴에서 못 보는 것들을 해야 의미가 있습니다:

1. Centroidal/SRBD MPC (강력 추천)

전체 18+ DOF는 너무 큼. Single Rigid Body Dynamics로 단순화
상태: COM 위치/속도, base orientation/angular velocity (12차원)
입력: 각 발의 GRF (3D × contact 발 개수)
→ 실제 MIT Cheetah, ANYmal에서 쓰는 방식
2. Contact schedule이 주어진 MPC

Gait (trot, walk)을 미리 정해놓고 contact timing은 fix
이게 일반적인 quadruped MPC. Mixed-integer 안 풀어도 됨
3. Whole-body QP와의 계층 구조

High-level: SRBD-MPC가 GRF와 base trajectory 계산 (50Hz)
Low-level: WBC가 joint torque로 변환 (1kHz)
→ 실제 로봇 제어 아키텍처 체험
4. Real-time 이슈

카트폴에선 안 보이는 것: solve time이 dt보다 길면? Warm-starting, RTI scheme