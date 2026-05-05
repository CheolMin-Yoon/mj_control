#### Mujoco, Pinocchio 시뮬레이션 프레임워크 (`mj_sim`)

이 레포지토리는 MuJoCo와 Pinocchio를 활용한 로봇 제어 프레임워크입니다. `mj_sim` 폴더 내에 로봇의 종류에 따라 모듈화되어 있습니다.

#### 환경 활성화 및 설치
conda env create -f environment.yml
conda activate mj


---

#### 작성원칙

##### 명명

1. **`compute_*`** — 호출 시점에 수치 연산을 새로 수행. 입력 → 출력의 함수형 계산.
   *Why*: 호출 비용이 있다는 신호. 루프에서 함부로 부르면 안 됨을 호출자에게 알림.
   *예*: `compute_DLS`, `compute_rpy_world`, `compute_bezier_trajectory`.

2. **`get_*`** — 인자를 받아 내부/외부 라이브러리에서 조회. 이미 계산된 값을 키로 꺼냄.
   *Why*: Pinocchio/MuJoCo 자체가 `getFrameJacobian`, `mj_contactForce` 패턴을 사용 → wrapping 컨벤션 일치.
   ⚠️ `set_*` 짝으로 만들지 말 것. setter는 property setter 또는 직접 속성 할당.
   *예*: `get_J_world(key)`, `get_hip_offset(leg)`, `get_foot_contact_state()`.

3. **`@property`** — 인자 없는 단일 값 노출. **싼 access + side-effect free**.
   *Why*: 호출자가 변수처럼 쓰므로 비싸거나 부작용 있으면 절대 안 됨. 비싼 건 `compute_*`로, 부작용 있는 건 동사형 메서드로.
   *예*: `wrapper.M`, `wrapper.com_pos_world`, `wrapper.J_com`.

4. **부작용 있는 동작은 동사형 메서드** — prefix 없이 동사 그대로.
   *예*: `update_model`, `register_foot_geoms`, `step`, `reset_yaw_unwrap`.

5. **시각화/플롯 prefix**:
   - matplotlib 함수 → `plot_*` (`plot_helpers.py`)
   - MuJoCo viewer 오버레이 → `draw_*` (`Visualizer` 메서드)

##### 변수 suffix

6. **`_curr` / `_des` / `_err`** 셋만 사용 (현재값 / 목표값 / 오차).
   - **suffix**로 통일 (prefix 금지). 같은 물리량(`pos_*`)이 grep·정렬·자동완성 시 묶임.
   - `target` / `desired` / `reference` 섞지 말고 `_des` 하나로 — robotics 표준어.

7. **물리량 prefix**: `pos_`, `vel_`, `acc_`, `tau_`, `F_`, `J_`, `M_`, `R_`, `q_`, `dq_`.
   **좌표계 suffix**: `_world` / `_body` / `_base`.
   *예*: `pos_curr`, `vel_des`, `pos_err`, `com_pos_world`, `R_body_to_world`.

##### 주석

8. **카테고리 헤더는 한국어, 번호 X**. 위·아래 빈 줄 1개로 경계.

   ```python
   # 상태

   @property
   def q(self):  ...
   @property
   def dq(self): ...

   # 동역학

   @property
   def M(self):     ...
   @property
   def M_inv(self): ...
   ```

9. **인라인 주석은 "왜"만**. 코드가 이미 말하는 "무엇"은 쓰지 않음.
   *예외*: 비표준 변환, 매직넘버, 외부 컨벤션 차이 (MJ vs Pin quat 순서 등).

10. **Docstring은 1줄 또는 생략**. 시그니처+이름이 자명하면 안 씀. 단위·shape·side effect가 모호한 경우만 짧게:

    ```python
    def compute_DLS(self, twist_des, twist_err, J):
        """V_des = twist_des + Kp·err → dq_des (na,)"""
    ```

##### 파일 구조

11. **클래스 내부 멤버 순서**:
    1. `__init__`
    2. 핵심 update/step (있다면) — `update_model`, `step`
    3. `@property` (카테고리별 묶음)
    4. 메서드 (`compute_*` → `get_*` → 변환 → 유틸 순)

    *Why*: 위에서 아래로 "초기화 → 매 step 동작 → 노출 상태 → 보조 연산"의 자연스러운 흐름.

12. **모듈 상수는 import 직후 상단**. 클래스 위에 둠.
    *예*: `G1_FRAMES`, `LEG_KEYS`, `LEG_IDX`.

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
- **MPC**:

---

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
