### Manipulator

UR5e / Franka Panda 기반 매니퓰레이터 제어. `core` (상태·동역학) → `control` (제어기) → `utils` (시각화·로깅) 계층.

#### 패키지 구조

##### [`core/`](core/) — 로봇 상태와 동역학
- [`fixed_base_robot_state.py`](core/fixed_base_robot_state.py): `q`, `dq`, `pos_curr` 등 상태 컨테이너 (fixed base용)
- [`pinocchio_wrapper.py`](core/pinocchio_wrapper.py): Pinocchio 모델 래퍼. `M`, `J_world`, `com_pos_world` 등 노출
- [`mujoco_kernel.py`](core/mujoco_kernel.py): MuJoCo step 루프, actuator 매핑

##### [`control/`](control/) — 제어기
- [`task_space_controller.py`](control/task_space_controller.py): DLS / impedance 등 task-space 제어
- [`model_pred_controller.py`](control/model_pred_controller.py): MPC
- [`trajectory_generator.py`](control/trajectory_generator.py): PTP (cubic/quintic), spline 궤적

##### [`utils/`](utils/) — 보조
- [`visualizer.py`](utils/visualizer.py): MuJoCo viewer 오버레이 (`draw_*`)
- [`plot_helpers.py`](utils/plot_helpers.py): matplotlib 플롯 (`plot_*`)
- [`data_logger.py`](utils/data_logger.py): 시뮬레이션 데이터 로깅
- [`sim_scheduler.py`](utils/sim_scheduler.py): 멀티레이트 스케줄링

#### Refactored Notebooks
모듈화된 패키지 사용 예시.
- [`ur5e_dls_remake.ipynb`](ur5e_dls_remake.ipynb): UR5e DLS 제어
- [`ur5e_impedance_remake.ipynb`](ur5e_impedance_remake.ipynb): UR5e Impedance 제어
- [`panda_impedance.ipynb`](panda_impedance.ipynb): Franka Panda Impedance 제어

#### Jupyter Study Files ([`jupyter_study_files/`](jupyter_study_files/))
초기 학습·기능 테스트용. 모듈화 이전의 한 노트북짜리 구현.

| 파일 | 내용 |
|---|---|
| `ur5e_mj_dls.ipynb` | 순수 MuJoCo DLS |
| `ur5e_mj2pin_pure_dls.ipynb` | DLS + Pinocchio |
| `ur5e_mj2pin_PTP.ipynb` | PTP 궤적 (quintic, cubic) |
| `ur5e_mj2pin_Spline.ipynb` | waypoints 궤적 (cubic) |
| `ur5e_mj2pin_Spline_6D.ipynb` | cubic + 6D control |
| `ur5e_mj2pin_ik_qp.ipynb` | IK를 QP로 |
| `ur5e_mj2pin_Spline_6D_RRT.ipynb` | DLS + RRT* |
| `ur5e_mj2pin_impedance.ipynb` | OCS impedance |

새 로봇 추가 가이드는 루트 [`README.md`](../../README.md#how-to-make-new) 참고.
