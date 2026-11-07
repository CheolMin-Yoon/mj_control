"""Cartpole upright 선형화 + 연속시간 LQR 안정화 데모"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.control import lqr_gain  # noqa: E402
from source.dynamics import CartpoleDynamics  # noqa: E402
from source.robots.cartpole.constants import (  # noqa: E402
    CARTPOLE_ACTUATED_JOINT_NAMES,
    CARTPOLE_POLE_LENGTH,
    CARTPOLE_SCENE_XML,
)
from source.sim import DataLogger, MujocoKernel, SimScheduler  # noqa: E402

# timestep 0.01 s → 제어도 매 physics step
CTRL_HZ = 100


def main():
    kernel = MujocoKernel(CARTPOLE_SCENE_XML, CARTPOLE_ACTUATED_JOINT_NAMES)
    model, data = kernel.model, kernel.data
    m_c = model.body_mass[model.body("cart").id]
    m_p = model.body_mass[model.body("pole").id]
    gear = model.actuator_gear[0, 0]
    print(f"m_c = {m_c:.4f} kg, m_p = {m_p:.4f} kg, L = {CARTPOLE_POLE_LENGTH:.4f} m, gear = {gear:.1f}")

    dyn = CartpoleDynamics(m_c=m_c, m_p=m_p, L=CARTPOLE_POLE_LENGTH)
    x_eq = np.zeros(4)
    A, B = dyn.linearize(x_eq, 0.0)
    # motor: force = ctrl × gear → ctrl 단위 B
    K = lqr_gain(A, B * gear, Q=np.diag([100.0, 1.0, 1.0, 1.0]), R=np.array([[1.0]]))

    # keyframe은 hanging_down → upright에서 pole 0.1 rad 교란으로 시작
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.1
    mujoco.mj_forward(model, data)

    logger = DataLogger()

    def on_control(t):
        x = np.concatenate([kernel.qpos, kernel.qvel])
        u = -K @ (x - x_eq)
        kernel.apply_joint_torque(u)
        logger.log(times=t, x=x, u=u)

    SimScheduler(model, data, ctrl_hz=CTRL_HZ).run(on_control, duration=100.0)

    times, x, u = logger.get("times"), logger.get("x"), logger.get("u")[:, 0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, x[:, 0], label="cart_pos")
    axes[0].set_ylabel("cart pos [m]")
    axes[1].plot(times, x[:, 1], label="pole_angle", color="orange")
    axes[1].set_ylabel("pole angle [rad]")
    axes[2].plot(times, u, label="u (ctrl)", color="green")
    axes[2].set_ylabel("control input")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.axhline(0, color="k", linestyle="--", linewidth=0.7)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Cartpole LQR")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
