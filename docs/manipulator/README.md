# Manipulator 데모

UR5e·Panda 고정 기반 팔의 작업공간 추종. 조립은 `scripts/`, 부품은 `source/{sim,dynamics,control,solvers}`다.

## 1. 실행 흐름

```
MujocoKernel(scene, joint_names)       qpos·dof·actuator 주소를 관절 순서로 매핑
    │ joint_pos·joint_vel
    ▼
PinocchioWrapper | MujocoWrapper       update_model → mass_matrix_inv, bias_forces, jacobian_w, frame_state_w
    │
    ▼
EETracking.sense(t)                    TrajectoryGenerator 목표 + 현재 frame 상태 → TaskSpaceController.tracking_error
    │ pose_err, twist_err
    ▼
script on_control                      impedance_torque | dls_joint_vel | dls_qp_cost + OSQPSolver
    │ apply_joint_torque | apply_joint_pos + apply_joint_force
    ▼
SimScheduler.run                       physics timestep, ctrl_hz, render_hz, real-time pacing
```

## 2. 데모

| Script | 장면 actuator | 제어 | 주기 |
| --- | --- | --- | --- |
| `panda_impedance.py` | motor | OSC 임피던스, Pinocchio(wn 10) 또는 `--dynamics mujoco`(wn 20) | 500 Hz |
| `ur5e_impedance.py` | motor | OSC 임피던스, 위치·자세 waypoint 7개 | 500 Hz |
| `ur5e_dls.py` | position | DLS 속도 IK 1-step 적분, bias feedforward off → on 두 번 | 200 Hz |
| `ur5e_qp.py` | position | DLS IK를 OSQP QP로, 관절 속도·위치 한계 | 100 Hz |

로그 신호는 `EETracking.log`가 `ee_pos_des_w`, `ee_pos_w`, `ee_lin_vel_des_w`, `ee_lin_vel_w`, `q`, `dq`, `ddq`로 남기고
`plot_tracking_report`가 그린다.

## 3. 제어식

- 오차: `pose_err = [p_des − p; log3(R_des Rᵀ)]`, `twist_err = V_des − V`, 모두 world 축
- OSC: `τ = Jᵀ Λ a_des + bias`, `Λ = (J M⁻¹ Jᵀ + λ²I)⁻¹`, `a_des = kp·pose_err + kd·twist_err`. `kp = wn²`, `kd = 2·wn`는
  가속도 PD 이득이고 물리 강성이 아니며, J̇q̇ 보상은 없다
- DLS: `dq = Jᵀ(JJᵀ + λ²I)⁻¹(V_des + kp·pose_err)`, QP는 같은 목적함수 `½dqᵀ(JᵀJ + λ²I)dq − (Jᵀ V)ᵀdq`에 box 제약

## 4. 알려진 제약

- Panda Pinocchio 모델(`panda_nohand.xml`)은 손이 없어 MuJoCo보다 1.9 kg 가볍고, keyframe 중력 토크가 joint2에서
  4.1 N·m 모자란다 → Pinocchio 경로는 정상상태 오차가 약 2 cm 남고 `--dynamics mujoco` 경로는 약 1 mm다
- `MujocoWrapper.update_model`은 제어 관절 q·dq를 `data`에 다시 쓴다. 제어 관절이 아닌 dof(그리퍼)는 고정값으로 취급한다
