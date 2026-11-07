# mj_control

MuJoCo 시뮬레이션과 Pinocchio 모델 함수로 고정 기반 로봇 제어기를 구현한다. 카트폴(LQR·MPC 기초), 매니퓰레이터(UR5e·Panda의 DLS·OSC
임피던스·IK QP), RB-Y1 양팔 SE(3) 임피던스를 다룬다.

## 설치

```bash
uv venv ~/.venvs/mj_control --python 3.14
source ~/.venvs/mj_control/bin/activate
uv sync --active
```

## 배치

```
mj_control/
├── source/
│   ├── sim/                  # MujocoKernel, SimScheduler, Visualizer, DataLogger, plot
│   ├── dynamics/             # PinocchioWrapper, MujocoWrapper, CartpoleDynamics, 이산화
│   ├── control/              # TaskSpaceController, TrajectoryGenerator, EETracking, lqr_gain
│   ├── solvers/              # OSQPSolver
│   └── robots/<robot>/       # constants.py, xmls/(MJCF·URDF·mesh, scene*.xml)
├── scripts/                  # viewer 실행 진입점
├── tests/                    # headless 검증
└── docs/                     # 작성원칙, 데모 문서
```

## 실행

저장소 루트에서 `~/.venvs/mj_control/bin/python scripts/<script>.py`. viewer 창을 닫으면 결과 플롯이 뜬다.

| Script | 내용 |
| --- | --- |
| `cartpole_lqr.py` | 카트폴 upright 선형화 + 연속시간 LQR |
| `indy7.py` | Indy7 v3 보기, home 자세에서 중력·Coriolis 보상으로 유지, TCP 축 표시 |
| `indy7_impedance.py` | Indy7 v3 Lie group 임피던스(Kim et al. TRO 2025), `--experiment push`·`rendering`·`half_turn` |
| `codegen_indy7_se3.py` | Indy7 v3 SE(3) 임피던스 항 CUDA 생성(FP64·FP32), 4096 lane GPU 검증 |
| `panda_impedance.py` | Panda OSC 임피던스, `--dynamics mujoco`면 MuJoCo API 동역학만 |
| `ur5e_impedance.py` | UR5e OSC 임피던스, 위치·자세 waypoint |
| `ur5e_dls.py` | UR5e DLS 속도 IK, bias feedforward off → on 두 번 실행 |
| `ur5e_qp.py` | UR5e DLS IK를 OSQP QP로, 관절 속도·위치 제약 |

장면만 보려면 `python -m mujoco.viewer --mjcf=source/robots/<robot>/xmls/<scene>.xml`.

## 문서

- [작성원칙](docs/작성원칙.md): 배치·명명·주석·lint 계약
- [manipulator](docs/manipulator/README.md), [cartpole](docs/cartpole/README.md), [rby1](docs/rby1/README.md)
- [Indy7 일반형·v3 자산](docs/indy7/README.md): 공식 출처, torque MJCF·URDF, frame·관절 순서

## 참고

- https://github.com/joonhyung-lee/mujoco-robotics-usage
- https://github.com/stack-of-tasks/pinocchio
- https://github.com/ozkannceylan/mujoco-robotics-lab
- https://github.com/Shunichi09/PythonLinearNonlinearControl
