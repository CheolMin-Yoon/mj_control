"""matplotlib 추종 결과 plot, 모두 non-blocking 뒤 `hold_until_all_fig_closed`로 대기"""

import matplotlib.pyplot as plt
import numpy as np


def plot_ee_tracking(times, pos_des, pos, title="EE Tracking"):
    """축별 위치 오차 pos_des - pos, 입력 (N, 3)"""
    err = np.asarray(pos_des) - np.asarray(pos)
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(title, fontsize=14)
    for i, (ax, label, color) in enumerate(zip(axes, "XYZ", "brg", strict=True)):
        ax.plot(times, err[:, i], color=color, linewidth=1)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{label} Error [m]")
        ax.grid(True)
    axes[-1].set_xlabel("Time [s]")
    _show()


def plot_velocity_tracking(times, vel_des, vel, title="Velocity Tracking"):
    """목표·실제와 오차 2단, 입력 (N,) 또는 (N, k)"""
    vel_des = np.asarray(vel_des).reshape(len(times), -1)
    vel = np.asarray(vel).reshape(len(times), -1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=14)
    for i in range(vel_des.shape[1]):
        ax1.plot(times, vel_des[:, i], "--", label=f"Desired[{i}]")
        ax1.plot(times, vel[:, i], "-", alpha=0.7, label=f"Actual[{i}]")
        ax2.plot(times, vel_des[:, i] - vel[:, i], "-", label=f"Error[{i}]")
    ax1.set_ylabel("Velocity [m/s]")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Error [m/s]")
    ax2.set_xlabel("Time [s]")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    _show()


def plot_joint_state(times, q, dq, ddq, joint_names, title="Joint State"):
    """관절 위치·속도·가속도 3단, 입력 (N, num_joints), 가속도 축은 첫 10 step 제외 범위"""
    q, dq, ddq = np.asarray(q), np.asarray(dq), np.asarray(ddq)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14)
    units = ("Joint Position [rad]", "Joint Velocity [rad/s]", "Joint Acceleration [rad/s²]")
    for ax, values, unit in zip(axes, (q, dq, ddq), units, strict=True):
        for i, name in enumerate(joint_names):
            ax.plot(times, values[:, i], label=name)
        ax.set_ylabel(unit)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(True)
    axes[-1].set_xlabel("Time [s]")

    # 초기 과도 가속도가 축 범위를 지배하지 않게 첫 10 step 제외
    if len(ddq) > 10:
        y_min, y_max = ddq[10:].min(), ddq[10:].max()
        margin = (y_max - y_min) * 0.1 or 1.0
        axes[2].set_ylim(y_min - margin, y_max + margin)
    _show()


def plot_3d_trajectory(pos, waypoints=None, title="3D Trajectory"):
    """3D 경로와 waypoint, 등비율 축, pos (N, 3)"""
    pos = np.asarray(pos)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "g-", linewidth=2, label="Trajectory")
    ax.scatter(*pos[0], c="b", s=50, label="Start")
    ax.scatter(*pos[-1], c="k", s=50, label="End")
    points = pos
    if waypoints is not None:
        waypoints = np.asarray(waypoints)
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="r", s=60, label="Waypoints")
        points = np.vstack([pos, waypoints])

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max((points.max(axis=0) - points.min(axis=0)).max() / 2.0, 1e-9) * 1.05
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    ax.legend(loc="best")
    _show()


def summarize_tracking_error(times, err, labels=("X", "Y", "Z"), plot_labels=("X", "Y", "Z"), title="Position Error"):
    """축별 max·min·RMS 출력과 선택 축 plot, err (N, len(labels))"""
    times, err = np.asarray(times), np.asarray(err)
    for i, label in enumerate(labels):
        e = err[:, i]
        i_max, i_min = int(np.argmax(e)), int(np.argmin(e))
        print(
            f"{label}: max = {e[i_max]:+.4f} m @ t={times[i_max]:.3f}s  |  "
            f"min = {e[i_min]:+.4f} m @ t={times[i_min]:.3f}s  |  "
            f"|max| = {np.abs(e).max():.4f} m  |  RMS = {np.sqrt((e**2).mean()):.4f} m"
        )

    colors = {"X": "red", "Y": "green", "Z": "blue"}
    plt.figure(figsize=(7, 4))
    for label in plot_labels:
        e = err[:, labels.index(label)]
        color = colors.get(label)
        i_max, i_min = int(np.argmax(e)), int(np.argmin(e))
        plt.plot(times, e, color=color, linewidth=1.2, label=f"{label} error")
        plt.scatter([times[i_max], times[i_min]], [e[i_max], e[i_min]], color=color, s=40, zorder=5)
        plt.annotate(
            f"max {e[i_max]:+.3f}",
            (times[i_max], e[i_max]),
            textcoords="offset points",
            xytext=(5, 5),
            color=color,
            fontsize=8,
        )
        plt.annotate(
            f"min {e[i_min]:+.3f}",
            (times[i_min], e[i_min]),
            textcoords="offset points",
            xytext=(5, -10),
            color=color,
            fontsize=8,
        )
    plt.axhline(0, color="k", linestyle="--", linewidth=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    _show()


def plot_tracking_report(logger, waypoints, joint_names, title, plot_labels=("X", "Y", "Z")):
    """`EETracking.log` 신호로 3D 경로·위치 오차·선속도·관절 상태·오차 통계 plot"""
    times = logger.get("times")
    pos_des, pos = logger.get("ee_pos_des_w"), logger.get("ee_pos_w")
    plot_3d_trajectory(pos, waypoints=waypoints, title=f"{title}: EE 3D Trajectory")
    plot_ee_tracking(times, pos_des, pos, title=f"{title}: EE Position Error")
    plot_velocity_tracking(
        times,
        logger.get("ee_lin_vel_des_w"),
        logger.get("ee_lin_vel_w"),
        title=f"{title}: EE Linear Velocity",
    )
    plot_joint_state(
        times,
        logger.get("q"),
        logger.get("dq"),
        logger.get("ddq"),
        joint_names,
        title=f"{title}: Joint States",
    )
    summarize_tracking_error(times, pos_des - pos, plot_labels=plot_labels, title=f"{title}: Position Error")


def hold_until_all_fig_closed():
    plt.show()


def _show():
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
