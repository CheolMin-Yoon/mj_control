#### Mujoco, Pinocchio 시뮬레이션 프레임워크 (`mj_sim`)

이 레포지토리는 MuJoCo와 Pinocchio를 활용한 로봇 제어 프레임워크입니다. `mj_sim` 폴더 내에 로봇의 종류에 따라 모듈화되어 있습니다.

#### 환경 활성화 및 설치
```bash
conda env create -f environment.yml
conda activate mj
```

---

### 1. Manipulator (`mj_sim/manipulator/`)

#### Refactored Scripts
기존 스터디 파일들을 `core`, `control`, `utils` 패키지로 모듈화하여 재작성한 버전입니다.
- `ur5e_dls_remake.ipynb`: DLS 제어 기반 Remake 버전
- `ur5e_impedance_remake.ipynb`: Impedance 제어 기반 Remake 버전

#### Jupyter Study Files (`jupyter_study_files/`)
초기 학습 및 기능 테스트를 위한 스크립트들입니다.
- **순수 무조코**: `ur5e_mj_dls.ipynb`
- **순수 DLS + pinocchio**: `ur5e_mj2pin_pure_dls.ipynb`
- **PTP 궤적 quintic, cubic**: `ur5e_mj2pin_PTP.ipynb`
- **waypoints 궤적 cubic**: `ur5e_mj2pin_Spline.ipynb`
- **cubic + 6D control**: `ur5e_mj2pin_Spline_6D.ipynb`
- **IK를 QP로 변경**: `ur5e_mj2pin_ik_qp.ipynb`
- **DLS + RRT***: `ur5e_mj2pin_Spline_6D_RRT.ipynb`
- **OCS impedance control**: `ur5e_mj2pin_impedance.ipynb`

---

### 2. Mobile (`mj_sim/mobile/`)

- **LQR error dynamics**: `husky_LQR_tracking.ipynb`

---

### 3. CartPole 

- iLQR + socu 
- quadruped robot wbc dynamics iLQR + socu

### How to Make New
새로운 로봇이나 시뮬레이션 환경을 구성할 때의 가이드입니다.

1. `core`, `utils` 패키지를 복사한다.
2. **State container** (예: `FloatingBaseRobotState`)
   - fixed base / floating base에 따라 컨테이너 선택
   - `NQ`, `NV` 매직넘버를 새 로봇 DOF에 맞게 수정
   - partition view (`left_leg_angle` 등) 슬라이스 인덱스 재계산
3. **Pinocchio wrapper**
   - 모듈 상단의 `FRAMES` dict — 새 URDF의 link 이름으로 교체
   - `LEG_KEYS` (있으면) — 새 로봇의 다리 구성에 맞춰 수정
   - `__init__`의 `hip_placements` / `shoulder_placement` 등 캐싱 로직 재검토
4. **MuJoCo kernel**
   - keyframe 인덱스 / actuator-joint 매핑 (`joint_names_pin_order`) 확인
   - `register_foot_geoms`에 전달할 geom/body 이름 점검
5. **컨벤션 준수**: 위 작성원칙대로 명명 / 주석 / 파일 구조.

---

### 참고 자료들
- https://github.com/joonhyung-lee/mujoco-robotics-usage
- https://github.com/stack-of-tasks/pinocchio
- https://github.com/ozkannceylan/mujoco-robotics-lab
- https://github.com/elijah-waichong-chan/go2-convex-mpc
- https://github.com/Shunichi09/PythonLinearNonlinearControl

---

### To Do Lists
1. Introduction to Humanoid Robotics — Kajita ✅
2. Modern Robotics — Kevin M. Lynch, Frank C. Park ✅
3. Convex Optimization — Stephen Boyd — ~ing
4. Underactuated Robotics — Russ Tedrake
5. Convex Model Predictive Control — MIT

---
