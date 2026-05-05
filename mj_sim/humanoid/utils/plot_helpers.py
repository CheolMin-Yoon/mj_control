import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D projection 등록용)


# 공통 스타일
_TITLE_FS  = 14
_AX_LABEL  = {'xtime': 'Time [s]'}
_LEG_DES   = 'Desired'
_LEG_CURR  = 'Current'
_LEG_ERR   = 'Error'


def _finalize(block, close=False):
    plt.tight_layout()
    plt.show(block=block)
    # inline 백엔드(notebook)는 pause 불필요 — 일부 GUI 백엔드에서 busy spin 회피
    if 'inline' not in matplotlib.get_backend().lower():
        plt.pause(0.001)
    if close:
        plt.close()


def plot_ee_tracking(times, pos_des, pos_curr, title="EE Tracking", block=False):
    """3축 위치 오차 plot (XYZ 각각 한 줄)
    pos_des, pos_curr: shape (N, 3)
    """
    times    = np.asarray(times)
    pos_des  = np.asarray(pos_des)
    pos_curr = np.asarray(pos_curr)
    pos_err  = pos_des - pos_curr

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(title, fontsize=_TITLE_FS)
    labels = ['X', 'Y', 'Z']
    colors = ['b', 'r', 'g']

    for i, (ax, lab, c) in enumerate(zip(axes, labels, colors)):
        ax.plot(times, pos_err[:, i], color=c, linewidth=1)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_ylabel(f'{lab} Error [m]')
        ax.grid(True)
    axes[-1].set_xlabel(_AX_LABEL['xtime'])

    _finalize(block)


def plot_velocity_tracking(times, vel_des, vel_curr, title="Velocity Tracking", block=False):
    """Desired vs Current + Error 2단 plot
    vel_des, vel_curr: shape (N,) 스칼라(norm) 또는 (N, k) 다차원
    """
    times    = np.asarray(times)
    vel_des  = np.asarray(vel_des)
    vel_curr = np.asarray(vel_curr)
    vel_err  = vel_des - vel_curr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=_TITLE_FS)

    if vel_des.ndim == 1:
        ax1.plot(times, vel_des,  'k--', label=_LEG_DES)
        ax1.plot(times, vel_curr, 'b-', alpha=0.7, label=_LEG_CURR)
        ax2.plot(times, vel_err,  'r-', label=_LEG_ERR)
    else:
        for i in range(vel_des.shape[1]):
            ax1.plot(times, vel_des[:, i],  '--', label=f'{_LEG_DES}[{i}]')
            ax1.plot(times, vel_curr[:, i], '-', alpha=0.7, label=f'{_LEG_CURR}[{i}]')
            ax2.plot(times, vel_err[:, i],  '-', label=f'{_LEG_ERR}[{i}]')

    ax1.set_ylabel('Velocity [m/s]')
    ax1.legend(loc='upper right'); ax1.grid(True)

    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Velocity Error [m/s]')
    ax2.set_xlabel(_AX_LABEL['xtime'])
    ax2.legend(loc='upper right'); ax2.grid(True)

    _finalize(block)


def plot_joint_state(times, q, dq, joint_names=None, title="Joint State", block=False):
    """관절 위치/속도 2단 plot. G1 29DOF 호환.
    q, dq: shape (N, n_joints)
    """
    times = np.asarray(times)
    q     = np.asarray(q)
    dq    = np.asarray(dq)
    n     = q.shape[1]
    if joint_names is None:
        joint_names = [f'J{i+1}' for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=_TITLE_FS)

    ncol = 4 if n > 12 else 3

    for i in range(n):
        ax1.plot(times, q[:, i], label=joint_names[i])
    ax1.set_ylabel('Joint Position [rad]')
    ax1.legend(loc='upper right', ncol=ncol, fontsize=8)
    ax1.grid(True)

    for i in range(n):
        ax2.plot(times, dq[:, i], label=joint_names[i])
    ax2.set_ylabel('Joint Velocity [rad/s]')
    ax2.set_xlabel(_AX_LABEL['xtime'])
    ax2.legend(loc='upper right', ncol=ncol, fontsize=8)
    ax2.grid(True)

    _finalize(block)


def plot_3d_trajectory(pos_log, waypoints=None, equal_aspect=True,
                       title="3D Trajectory", block=False):
    """3D 궤적 plot. base가 움직이는 휴머노이드용 등비율 박스 지원.
    pos_log:   shape (N, 3) 강제. (3, N)이면 호출자가 .T로 전치 후 전달.
    waypoints: shape (M, 3) optional.
    """
    pos = np.asarray(pos_log)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos_log shape (N, 3) 필요. got {pos.shape}")

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'g-', linewidth=2, label='Trajectory')
    ax.scatter(pos[0, 0],  pos[0, 1],  pos[0, 2],  c='b', s=50, label='Start')
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='k', s=50, label='End')

    if waypoints is not None:
        wp = np.asarray(waypoints)
        if wp.ndim != 2 or wp.shape[1] != 3:
            raise ValueError(f"waypoints shape (M, 3) 필요. got {wp.shape}")
        ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c='r', s=60, label='Waypoints')

    ax.set_title(title, fontsize=_TITLE_FS - 1)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')

    if equal_aspect:
        data = pos.T if waypoints is None else np.hstack([pos.T, wp.T])
        mins = data.min(axis=1); maxs = data.max(axis=1)
        ctr  = (mins + maxs) / 2.0
        r    = max((maxs - mins).max() / 2.0, 1e-9) * 1.05
        ax.set_xlim(ctr[0]-r, ctr[0]+r)
        ax.set_ylim(ctr[1]-r, ctr[1]+r)
        ax.set_zlim(ctr[2]-r, ctr[2]+r)
        ax.set_box_aspect([1, 1, 1])

    ax.grid(True); ax.legend(loc='best')
    _finalize(block)


def plot_solve_time(solve_ms, compute_ms, dt, hz,
                    title="MPC Iteration Stats", block=False):
    """MPC/QP solve time stacked bar + real-time budget line"""
    solve_ms   = np.asarray(solve_ms)
    compute_ms = np.asarray(compute_ms)
    total_ms   = solve_ms + compute_ms
    iters      = np.arange(len(solve_ms))
    budget_ms  = dt * 1e3

    _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(iters, compute_ms, label='Model Update [ms]')
    ax.bar(iters, solve_ms, bottom=compute_ms, label='QP Solve [ms]')
    ax.axhline(budget_ms, linestyle='--', linewidth=2.0,
               label=f'Real-Time Budget {hz} Hz ({budget_ms:.1f} ms)')

    ax.set_title(title, fontsize=_TITLE_FS)
    ax.set_xlabel('MPC Step', fontweight='bold')
    ax.set_ylabel('Time [ms]', fontweight='bold')
    ax.set_ylim(bottom=0)

    text_str = (
        f"Avg Model Update: {compute_ms.mean():.2f} ms\n"
        f"Avg QP Solve:     {solve_ms.mean():.2f} ms\n"
        f"Avg Cycle:        {total_ms.mean():.2f} ms"
    )
    ax.text(0.02, 0.7, text_str,
            transform=ax.transAxes, va='center', ha='left',
            bbox=dict(boxstyle='round', alpha=0.3))

    ax.legend()
    _finalize(block)


def plot_contact_schedule(times, contact_mask, force, dt,
                          leg_names=("L_foot", "R_foot"),
                          title="Contact Forces & Schedule", block=False):
    """발 접촉 스케줄(swing 음영) + 접촉력 ZOH plot.
    contact_mask: shape (n_legs, N), 1=stance / 0=swing
    force: shape (3*n_legs, N), 다리별 [fx, fy, fz]
    """
    contact_mask = np.asarray(contact_mask)
    force        = np.asarray(force)
    n_legs       = contact_mask.shape[0]
    N            = contact_mask.shape[1]
    assert force.shape[0] == 3 * n_legs, \
        f"force는 (3*n_legs, N) 형태여야 함. got {force.shape}"

    t_edges = np.linspace(0, N * dt, N + 1)

    fig, axes = plt.subplots(n_legs, 1, figsize=(10, 3 * n_legs), sharex=True)
    if n_legs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        fx, fy, fz = force[3*i:3*i+3]
        ax.stairs(fx, t_edges, label='fx')
        ax.stairs(fy, t_edges, label='fy')
        ax.stairs(fz, t_edges, label='fz', linewidth=2)

        swing = (contact_mask[i] == 0)
        for k in np.flatnonzero(swing):
            ax.axvspan(t_edges[k], t_edges[k+1], alpha=0.15,
                       hatch='//', edgecolor='none')

        ax.set_ylabel(f'{leg_names[i]} [N]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=3, fontsize=9)

    axes[-1].set_xlabel(_AX_LABEL['xtime'])
    fig.suptitle(title, fontsize=_TITLE_FS)
    _finalize(block)


def hold_until_all_fig_closed():
    """모든 figure가 닫힐 때까지 대기. non-blocking plot들 마지막에 호출."""
    plt.show()
