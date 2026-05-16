# %%
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin
from scipy.spatial import ConvexHull
from core    import FloatingBaseRobotState, Pinocchio_Wrapper, Mujoco_Kernel, LEG_KEYS, LEG_IDX
from control import (GaitScheduler, LegController, MotionPlanner,
                     SingleRigidBodyDynamics, OSQP_Solver, ProxQP_Solver)
from utils   import SimScheduler, Visualizer

URDF = str(_ROOT / 'models' / 'go2' / 'urdf' / 'go2_description.urdf')
PKG  = str(_ROOT / 'models' / 'go2')
MJCF = str(_ROOT / 'models' / 'scenes' / 'go2' / 'scene.xml')

# %%
def compute_reference_trajectory(x_current, pos_des_W, v_body_cmd, yaw_rate_cmd, z_target,
                                  N, dt, max_pos_error=0.1):
    """
    reference (go2-convex-mpc/com_trajectory.py) 방식 — 누적 desired position + clamp.

    x_current     : (13,) [rpy, p, omega, v, -1]
    pos_des_W     : (3,) in/out — 외부에서 유지되는 cumulative desired CoM 위치 (world).
                    매 호출마다 현재 위치 ±max_pos_error 안으로 in-place clamp.
    v_body_cmd    : (3,) body frame 속도 명령
    yaw_rate_cmd  : scalar
    z_target      : scalar — z는 매번 desired로 reset
    max_pos_error : robot이 따라잡을 수 있도록 desired가 robot에서 벗어나는 한계 [m]

    pos_des가 매 tick reset되지 않고 누적되므로 robot이 desired에 뒤처지면 MPC가
    그 offset만큼 추진력을 만들어 v_des를 추종한다 (steady-state position offset).
    """
    p_now    = x_current[3:6]
    yaw_curr = x_current[2]

    # pos_des → 현재 위치 ±max_pos_error 안으로 clamp (xy), z는 desired
    pos_des_W[0] = np.clip(pos_des_W[0], p_now[0] - max_pos_error, p_now[0] + max_pos_error)
    pos_des_W[1] = np.clip(pos_des_W[1], p_now[1] - max_pos_error, p_now[1] + max_pos_error)
    pos_des_W[2] = z_target

    # body → world (yaw rotation only)
    cy, sy = np.cos(yaw_curr), np.sin(yaw_curr)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    v_world = Rz @ v_body_cmd

    # horizon broadcasting — (3, N)
    t_vec     = (np.arange(N) + 1) * dt                            # (N,)
    pos_traj  = pos_des_W[:, None] + v_world[:, None] * t_vec      # (3, N)
    rpy_traj  = np.zeros((3, N))
    rpy_traj[2] = yaw_curr + yaw_rate_cmd * t_vec
    vel_traj  = np.broadcast_to(v_world[:, None], (3, N))          # constant
    omega_traj = np.zeros((3, N))
    omega_traj[2] = yaw_rate_cmd

    # stack into 13*N (rpy, p, omega, v, gravity=0 trick)
    X_ref = np.zeros(13 * N)
    for i in range(N):
        idx = i * 13
        X_ref[idx     : idx + 3 ] = rpy_traj[:, i]
        X_ref[idx + 3 : idx + 6 ] = pos_traj[:, i]
        X_ref[idx + 6 : idx + 9 ] = omega_traj[:, i]
        X_ref[idx + 9 : idx + 12] = vel_traj[:, i]
        X_ref[idx + 12]           = 0.0
    return X_ref

# %%
state    = FloatingBaseRobotState()
wrapper  = Pinocchio_Wrapper(URDF, PKG)
joint_names = [wrapper.model.names[i] for i in range(2, wrapper.model.njoints)]
kernel   = Mujoco_Kernel(MJCF, joint_names_pin_order=joint_names)
kernel.register_foot_geoms({"FL": "FL", "FR": "FR", "RL": "RL", "RR": "RR"})
gait     = GaitScheduler(gait_hz=3.0, duty=0.6)

# 속도 커맨드 (body frame)
VX_CMD       = -0.2   # m/s  전진
YAW_RATE_CMD = 0.   # rad/s
Z_TARGET     = 0.25  # m  서있는 높이

# MPC weight (go2-convex-mpc reference: COST_MATRIX_Q = diag([1,1,50, 10,20,1, 2,2,1, 1,1,1]))
# reference state [p, rpy, v, omega] → 우리 state [rpy, p, omega, v, gravity] 매핑
Q_MPC = np.diag([  1,   1,   50,   # roll/pitch/yaw  ← reference rpy
                  10,  20,   1,   # px/py/pz        ← reference p
                   1,   1,   1,   # omega           ← reference omega
                   2,   2,   1,   # vx/vy/vz        ← reference v
                   0])            # gravity trick → 가중치 0
R_MPC = 1e-5 * np.eye(12)

MU      = 0.8    # 마찰계수
FZ_MAX  = 660.0  # N

# multi-rate:  MPC 100Hz / OCS(=ctrl_tau) 500Hz / physics는 model.opt.timestep
DT_MPC = 0.01    # 100Hz
N_MPC  = int(gait.gait_period / DT_MPC)   # gait 한 cycle 기준 (reference 컨벤션)

motion_planner = MotionPlanner(wrapper)
leg_ctrl = LegController(wrapper, gait, motion_planner, kernel,
                         vx_des=VX_CMD, vy_des=0.0, yaw_rate_des=YAW_RATE_CMD)
dynamics = SingleRigidBodyDynamics(wrapper, Q_MPC, R_MPC, DT_MPC, N_MPC)
# constraint: friction cone (4-face) + fz_min per leg per step  → 5*4*N rows
solver   = ProxQP_Solver(n=12*N_MPC, m=5*4*N_MPC)

print("시스템 초기화 완료")
print(f"로봇 총 질량: {wrapper.m:.2f} kg")

# %% [markdown]
# Main Loop (상태 갱신, 제어, 뷰어, 시각화)

# %%
from control.motion_planner import HEIGHT_SWING

_setup_done   = False
_t_last_mpc   = -np.inf      # 첫 호출에서 무조건 MPC 풀게
_f_opt_cache  = None         # 직전 MPC 결과 (500Hz step에서 ZOH로 재사용)
_pos_des_W    = None         # cumulative desired CoM 위치 (첫 tick에 robot 위치로 init)

def _build_constraint(mask_h, N, mu, fz_max):
    """friction cone (4-face linearized) + fz bound, horizon별 contact schedule 반영."""
    A_leg = np.array([
        [ 1,  0, -mu],
        [-1,  0, -mu],
        [ 0,  1, -mu],
        [ 0, -1, -mu],
        [ 0,  0,  1 ],
    ])  # (5,3)

    n_f  = 12 * N
    n_c  = 5 * 4 * N
    A    = np.zeros((n_c, n_f))
    l    = np.full(n_c, -np.inf)
    u    = np.zeros(n_c)

    for k in range(N):
        mk = mask_h[k]
        for i in range(4):
            row = (k * 4 + i) * 5
            col = k * 12 + i * 3
            if mk[i] == 1:
                A[row:row+5, col:col+3] = A_leg
                l[row:row+4] = -np.inf
                u[row:row+4] = 0.0
                l[row+4]     = 10.0
                u[row+4]     = fz_max
            else:
                A[row:row+5, col:col+3] = np.eye(5, 3)
                l[row:row+5] = 0.0
                u[row:row+5] = 0.0
    return A, l, u


def _compute_foot_lever_horizon(pos_traj_W):
    """Horizon 동안 4발 lever arm (foot_world - base_traj_W). (3, 4, N).

    근사: 발 위치는 horizon 내내 현재 위치 고정 (touchdown은 단일 시점만 반영).
    base는 reference trajectory의 pos_traj_W를 따라 이동.
    swing leg의 lever는 force=0이라 결과 무관."""
    r_feet_W = wrapper.r_feet                             # (3, 4)
    # (3, 4, 1) - (3, 1, N) → (3, 4, N)
    return r_feet_W[:, :, None] - pos_traj_W[:, None, :]


def on_control(t):
    """500Hz로 호출. MPC는 100Hz로 throttle(DT_MPC 간격), 그 외는 ZOH."""
    global _setup_done, _t_last_mpc, _f_opt_cache, _pos_des_W

    kernel.update_robot_state(state)
    wrapper.update_model(state.q, state.dq)

    # ── MPC tick (100Hz) ─────────────────────────────────────
    if t - _t_last_mpc >= DT_MPC:
        rpy     = state.compute_rpy_world()
        p_com   = wrapper.p_com_W
        omega_W = wrapper.R_WB @ state.base_ang_vel
        v_com   = wrapper.v_com_W
        x_now   = np.concatenate([rpy, p_com, omega_W, v_com, [-1.0]])

        # 첫 tick: desired position을 현재 CoM 위치로 초기화
        if _pos_des_W is None:
            _pos_des_W = p_com.copy()

        v_body_cmd = np.array([VX_CMD, 0.0, 0.0])
        x_des = compute_reference_trajectory(x_now, _pos_des_W, v_body_cmd,
                                             YAW_RATE_CMD, Z_TARGET, N_MPC, DT_MPC)

        mask_horizon = gait.compute_mask_horizon(t, DT_MPC, N_MPC)
        dynamics.update_horizon(mask_horizon)   # r_feet_horizon=None → 현재 시점 r_skew만 사용
        H, g_vec = dynamics.compute_condensed_qp(x_now, x_des)

        A_c, l_c, u_c = _build_constraint(mask_horizon, N_MPC, MU, FZ_MAX)
        if not _setup_done:
            solver.setup(H, g_vec, A_c, l_c, u_c)
            _setup_done = True
        else:
            solver.update(H=H, q=g_vec, A=A_c, l=l_c, u=u_c)
        _f_opt_cache = solver.solve()
        _t_last_mpc  = t

    # ── OCS tick (500Hz) — stance: J^T·(-f), swing: KP/KD + Lambda FF + nle ──
    f_grf = _f_opt_cache[:12] if _f_opt_cache is not None else None
    tau   = leg_ctrl.compute_tau(t, f_grf, pos_des_W=_pos_des_W)
    kernel.ctrl_tau = tau

    if t < 0.02:
        print(f"t={t:.4f}  mask={gait.compute_gait_state(t)[0]}")
        print(f"  f_grf={None if f_grf is None else f_grf.round(2)}")
        print(f"  tau  ={tau.round(2)}")


def on_render(t, view):
    """매 render frame: Raibert touchdown 사각형 + swing 발 끝 궤적."""
    mask, _ = gait.compute_gait_state(t)
    eye3 = np.eye(3)

    for leg in LEG_KEYS:
        idx = LEG_IDX[leg]

        # 다음 발 위치 (Raibert touchdown) — 빨강 납작 박스
        view.draw_geom(
            mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.025, 0.002],
            pos=leg_ctrl._p_target[:, idx],
            mat=eye3,
            rgba=(1.0, 0.2, 0.2, 0.9),
        )

        # 발 끝 swing 궤적 — swing 중인 다리만 (시안색 line)
        if mask[idx] == 0:
            pts = [
                motion_planner.compute_bezier_trajectory(
                    phase,
                    leg_ctrl._p_takeoff[:, idx],
                    leg_ctrl._p_target[:, idx],
                    gait.swing_time,
                    HEIGHT_SWING,
                )[0]
                for phase in np.linspace(0.0, 1.0, 20)
            ]
            view.draw_connected_path(pts, rgba=(0.0, 0.8, 1.0, 1.0), thickness=0.003)

# %%
kernel.reset_to_keyframe()
sched = SimScheduler(kernel.model, kernel.data, ctrl_hz=500, render_hz=60)
sched.run(on_control=on_control, on_render=on_render)


