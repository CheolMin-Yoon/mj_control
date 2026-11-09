#### Mujoco, Pinocchio 시뮬레이션 프레임워크 (`mj_sim`)

이 레포지토리는 MuJoCo와 Pinocchio를 활용한 로봇 제어 프레임워크입니다. `mj_sim` 폴더 내에 로봇/주제별로 모듈화되어 있습니다.

Go2 제어 코드와 자산은 [go2-mjlab의 `legacy/mpc-mujoco`](https://github.com/CheolMin-Yoon/go2-mjlab/tree/legacy/mpc-mujoco)를 참조하세요.

#### 환경 활성화 및 설치
```bash
conda env create -f environment.yml
conda activate mj


# 모든 주피터 출력 삭제
nbstripout mj_sim/**/*.ipynb
```



#### 작성원칙
명명·주석·파일 구조 컨벤션은 [mj_sim/작성원칙.md](mj_sim/작성원칙.md)를 참고하세요.

#### 공통 유틸
- [mj_sim/spawn_mujoco.ipynb](mj_sim/spawn_mujoco.ipynb): MuJoCo 모델 스폰 및 viewer 띄우기 템플릿
- [mj_sim/casADi_test.ipynb](mj_sim/casADi_test.ipynb): CasADi 기반 NLP / QP / MPC 빌딩 블록 테스트

---

### 1. Manipulator ([mj_sim/manipulator/](mj_sim/manipulator/))

UR5e / Franka Panda 기반 매니퓰레이터. DLS, Impedance, PTP/Spline 궤적, IK-QP, RRT*, MPC 등.
세부 구조와 노트북 목록은 [mj_sim/manipulator/README.md](mj_sim/manipulator/README.md) 참고.

---

### 2. CartPole ([mj_sim/cartpole/](mj_sim/cartpole/))

상태 4차원·입력 1차원의 표준 벤치마크. 최적제어 이론을 손으로 디버깅하며 체화하는 단계.

- [LQR_구현.md](mj_sim/cartpole/LQR_구현.md): 평형점 선형화 기반 LQR, finite-horizon / TVLQR, DARE까지 단계별 구현 노트
- [MPC_구현.md](mj_sim/cartpole/MPC_구현.md): Linear/Nonlinear MPC, condensed vs sparse, horizon·terminal cost 영향 검증 노트

---

### How to Make New
새로운 로봇이나 시뮬레이션 환경을 구성할 때의 가이드입니다. (manipulator 패키지 기준)

1. `core`, `utils` 패키지를 복사한다.
2. **State container** (예: `FixedBaseRobotState`)
   - fixed base / floating base에 따라 컨테이너 선택
   - `NQ`, `NV` 매직넘버를 새 로봇 DOF에 맞게 수정
   - partition view 슬라이스 인덱스 재계산
3. **Pinocchio wrapper**
   - 모듈 상단의 `FRAMES` dict — 새 URDF의 link 이름으로 교체
   - `__init__`의 placement 캐싱 로직 재검토
4. **MuJoCo kernel**
   - keyframe 인덱스 / actuator-joint 매핑 (`joint_names_pin_order`) 확인
   - geom/body 이름 점검
5. **컨벤션 준수**: [mj_sim/작성원칙.md](mj_sim/작성원칙.md)대로 명명 / 주석 / 파일 구조.

---

### 참고 자료들
- https://github.com/joonhyung-lee/mujoco-robotics-usage
- https://github.com/stack-of-tasks/pinocchio
- https://github.com/ozkannceylan/mujoco-robotics-lab
- https://github.com/Shunichi09/PythonLinearNonlinearControl

---

### To Do Lists
1. Introduction to Humanoid Robotics — Kajita ✅
2. Modern Robotics — Kevin M. Lynch, Frank C. Park ✅
3. Convex Optimization — Stephen Boyd — ~ing
4. Underactuated Robotics — Russ Tedrake
5. Convex Model Predictive Control — MIT

---
