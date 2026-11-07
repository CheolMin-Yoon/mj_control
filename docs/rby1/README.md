# RB-Y1

RB-Y1 M 1.3의 베이스를 고정한 상체 20자유도(몸통 6 + 왼팔 7 + 오른팔 7)에서 두 손의 SE(3) 임피던스를 구현한다.
자산과 상수는 `source/robots/rby1/`, 양팔 제어기는 `source/control/`, 실행은 `scripts/`에 둔다(아직 없음).

계획·제어기 사양·확인 항목의 정본은 [rby1-mjlab `docs/plan.md`](/home/frlab/rby1-mjlab/docs/plan.md)다. MJCF 자산은
rby1-mjlab `source/robots/rby1/xmls/`에 있다.

## 양팔 확장 전 확인

- 이미 지원: `MujocoKernel`은 관절 이름으로 qpos·dof·actuator 주소를 따로 매핑하고, wrapper frame key는 임의 개수다
  (`{"left_hand": ..., "right_hand": ...}`)
- 필요: `TaskSpaceController`는 frame 하나의 6×6 Λ만 계산한다 → 결합 12×20 Jacobian과 nullspace 자세 작업
- 필요: OSC 가속 항과 중력·Coriolis 분리가 필요하면 wrapper에 `mass_matrix`·`generalized_gravity`·`jdot_velocity_w`를 더한다
  ([작성원칙 §3](../작성원칙.md#3-동역학-항))
- 모델 대조: Pinocchio(URDF)와 MuJoCo(MJCF)의 질량·중력 토크·손끝 위치를 같은 q에서 비교한다. Panda는 손 질량 누락으로 정상상태
  오차 약 2 cm가 남았다([manipulator §4](../manipulator/README.md#4-알려진-제약))
