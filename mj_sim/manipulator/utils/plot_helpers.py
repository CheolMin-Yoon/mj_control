import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D projection 등록용)


def plot_ee_tracking(times, desired, actual, title="EE Tracking", block=False):
    """3축 위치 오차 plot (XYZ 각각 한 줄)
    desired, actual: shape (N, 3)
    """
    times = np.asarray(times)
    desired = np.asarray(desired)
    actual = np.asarray(actual)
    err = desired - actual

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(title, fontsize=14)
    labels = ['X', 'Y', 'Z']
    colors = ['b', 'r', 'g']

    for i, (ax, lab, c) in enumerate(zip(axes, labels, colors)):
        ax.plot(times, err[:, i], color=c, linewidth=1)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_ylabel(f'{lab} Error [m]')
        ax.grid(True)
    axes[-1].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)


def plot_velocity_tracking(times, des_vel, act_vel, title="Velocity Tracking", block=False):
    """Desired vs Actual + Error 2단 plot
    des_vel, act_vel: shape (N,) 스칼라(norm) 또는 (N, k) 다차원
    """
    times = np.asarray(times)
    des = np.asarray(des_vel)
    act = np.asarray(act_vel)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=14)

    if des.ndim == 1:
        ax1.plot(times, des, 'k--', label='Desired')
        ax1.plot(times, act, 'b-', alpha=0.7, label='Actual')
        ax2.plot(times, des - act, 'r-', label='Error')
    else:
        for i in range(des.shape[1]):
            ax1.plot(times, des[:, i], '--', label=f'Desired[{i}]')
            ax1.plot(times, act[:, i], '-', alpha=0.7, label=f'Actual[{i}]')
            ax2.plot(times, des[:, i] - act[:, i], '-', label=f'Error[{i}]')

    ax1.set_ylabel('Velocity [m/s]')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Error [m/s]')
    ax2.set_xlabel('Time [s]')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)


def plot_joint_state(times, q, dq, ddq, joint_names=None, title="Joint State", block=False):
    """관절 위치/속도/가속도 3단 plot.
    q, dq, ddq: shape (N, n_joints)
    joint_names: 길이 n_joints 리스트
    """
    times = np.asarray(times)
    q = np.asarray(q)
    dq = np.asarray(dq)
    ddq = np.asarray(ddq)
    n = q.shape[1]
    if joint_names is None:
        joint_names = [f'J{i+1}' for i in range(n)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14)

    ncol = 4 if n > 12 else 3
    
    for i in range(n):
        ax1.plot(times, q[:, i], label=joint_names[i])
    ax1.set_ylabel('Joint Position [rad]')
    ax1.legend(loc='upper right', ncol=ncol, fontsize=8)
    ax1.grid(True)

    for i in range(n):
        ax2.plot(times, dq[:, i], label=joint_names[i])
    ax2.set_ylabel('Joint Velocity [rad/s]')
    ax2.set_xlabel('Time [s]')
    ax2.legend(loc='upper right', ncol=ncol, fontsize=8)
    ax2.grid(True)
    
    for i in range(n):
        ax3.plot(times, ddq[:, i], label=joint_names[i])
    ax3.set_ylabel('Joint Acceleration [rad/s²]')
    ax3.set_xlabel('Time [s]')
    ax3.legend(loc='upper right', ncol=ncol, fontsize=8)
    ax3.grid(True)
    
    if len(ddq) > 10:
        safe_ddq = ddq[10:] # 첫 10스텝(약 0.02초) 무시
        y_min, y_max = np.min(safe_ddq), np.max(safe_ddq)
        margin = (y_max - y_min) * 0.1 # 위아래 10% 여백
        # 만약 로봇이 전혀 안 움직여서 margin이 0이 되는 경우를 대비
        if margin == 0: margin = 1.0 
        ax3.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)


def plot_3d_trajectory(pos_log, waypoints=None, equal_aspect=True,
                       title="3D Trajectory", block=False):
    """3D 궤적 plot. base가 움직이는 휴머노이드용으로 등비율 박스 지원.
    pos_log: shape (N, 3) 또는 (3, N)
    waypoints: shape (M, 3) optional
    """
    pos = np.asarray(pos_log)
    if pos.shape[0] == 3 and pos.shape[1] != 3:
        pos = pos.T  # (3, N) -> (N, 3)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'g-', linewidth=2, label='Trajectory')
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='b', s=50, label='Start')
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='k', s=50, label='End')

    if waypoints is not None:
        wp = np.asarray(waypoints)
        if wp.shape[0] == 3 and wp.shape[1] != 3:
            wp = wp.T
        ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c='r', s=60, label='Waypoints')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')

    if equal_aspect:
        data = pos.T if waypoints is None else np.hstack([pos.T, wp.T])
        mins = data.min(axis=1); maxs = data.max(axis=1)
        ctr = (mins + maxs) / 2.0
        half = max((maxs - mins).max() / 2.0, 1e-9)
        pad = 1.05
        r = half * pad
        ax.set_xlim(ctr[0]-r, ctr[0]+r)
        ax.set_ylim(ctr[1]-r, ctr[1]+r)
        ax.set_zlim(ctr[2]-r, ctr[2]+r)
        ax.set_box_aspect([1, 1, 1])

    ax.grid(True)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)


def plot_solve_time(solve_ms, compute_ms, dt, hz, title="MPC Iteration Stats", block=False):
    """MPC/QP solve time stacked bar + real-time budget line"""
    solve_ms = np.asarray(solve_ms)
    compute_ms = np.asarray(compute_ms)
    total_ms = solve_ms + compute_ms
    iters = np.arange(len(solve_ms))
    budget_ms = dt * 1e3

    _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(iters, compute_ms, label='Model Update Time (ms)')
    ax.bar(iters, solve_ms, bottom=compute_ms, label='QP Solve Time (ms)')
    ax.axhline(budget_ms, linestyle='--', linewidth=2.0,
               label=f'Real-Time Budget {hz} Hz ({budget_ms:.1f} ms)')

    ax.set_xlabel('MPC Step', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_ylim(bottom=0)

    text_str = (
        f"Avg Model Update: {compute_ms.mean():.2f} ms\n"
        f"Avg QP Solve:     {solve_ms.mean():.2f} ms\n"
        f"Avg Cycle:        {total_ms.mean():.2f} ms"
    )
    ax.text(
        0.02, 0.7, text_str,
        transform=ax.transAxes,
        va='center', ha='left',
        bbox=dict(boxstyle='round', alpha=0.3)
    )

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)


def plot_contact_schedule(times, contact_mask, force, dt,
                          leg_names=("L_foot", "R_foot"), block=False):
    """발 접촉 스케줄(swing 음영) + 접촉력 ZOH plot.
    contact_mask: shape (n_legs, N), 1=stance / 0=swing
    force: shape (3*n_legs, N), 다리별 [fx, fy, fz] 순서
    """
    contact_mask = np.asarray(contact_mask)
    force = np.asarray(force)
    n_legs = contact_mask.shape[0]
    N = contact_mask.shape[1]
    assert force.shape[0] == 3 * n_legs, \
        f"force는 (3*n_legs, N) 형태여야 함. got {force.shape}"

    t_edges = np.linspace(0, N * dt, N + 1)

    fig, axes = plt.subplots(n_legs, 1, figsize=(10, 3 * n_legs), sharex=True)
    if n_legs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        fx = force[3*i + 0]
        fy = force[3*i + 1]
        fz = force[3*i + 2]
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

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Contact Forces & Schedule')
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)


def summarize_tracking_error(times, err, axes=('X', 'Y', 'Z'),
                             title="Position Error", plot=True,
                             plot_axes=('X', 'Z'), block=False):
    """축별 추종 오차 통계 (max/min/RMS) 계산 + 옵션으로 그래프.

    times: shape (N,)
    err:   shape (N, len(axes))  — 보통 target_pos - actual_pos
    axes:  통계 라벨 (3D 기본)
    plot_axes: 그래프에 표시할 축 부분집합 (마커·주석 포함)
    """
    times = np.asarray(times)
    err   = np.asarray(err)
    assert err.shape[1] == len(axes), f"err.shape[1]={err.shape[1]} ≠ axes={len(axes)}"

    # ── 통계 출력 ──
    stats = {}
    for i, lab in enumerate(axes):
        e = err[:, i]
        i_max, i_min = int(np.argmax(e)), int(np.argmin(e))
        stats[lab] = {
            'max':       float(e[i_max]),
            'max_t':     float(times[i_max]),
            'min':       float(e[i_min]),
            'min_t':     float(times[i_min]),
            'abs_max':   float(np.abs(e).max()),
            'rms':       float(np.sqrt((e**2).mean())),
        }
        print(f'{lab}: max = {stats[lab]["max"]:+.4f} m @ t={stats[lab]["max_t"]:.3f}s  |  '
              f'min = {stats[lab]["min"]:+.4f} m @ t={stats[lab]["min_t"]:.3f}s  |  '
              f'|max| = {stats[lab]["abs_max"]:.4f} m  |  RMS = {stats[lab]["rms"]:.4f} m')

    if not plot:
        return stats

    # ── 그래프 (선택 축) + max/min 마커 ──
    color_map = {'X': 'red', 'Y': 'green', 'Z': 'blue'}
    plt.figure(figsize=(7, 4))
    for lab in plot_axes:
        idx = axes.index(lab)
        e   = err[:, idx]
        c   = color_map.get(lab, None)
        plt.plot(times, e, color=c, linewidth=1.2, label=f'{lab} error')
        i_max = int(np.argmax(e)); i_min = int(np.argmin(e))
        plt.scatter([times[i_max], times[i_min]], [e[i_max], e[i_min]],
                    color=c, s=40, zorder=5)
        plt.annotate(f'max {e[i_max]:+.3f}', (times[i_max], e[i_max]),
                     textcoords='offset points', xytext=(5, 5),
                     color=c, fontsize=8)
        plt.annotate(f'min {e[i_min]:+.3f}', (times[i_min], e[i_min]),
                     textcoords='offset points', xytext=(5, -10),
                     color=c, fontsize=8)

    plt.axhline(0, color='k', linestyle='--', linewidth=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Position Error [m]')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)
    return stats


def hold_until_all_fig_closed():
    """모든 figure가 닫힐 때까지 대기. non-blocking plot들 마지막에 호출."""
    plt.show()
