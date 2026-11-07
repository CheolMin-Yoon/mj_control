"""Indy7 v3 보기, home keyframe에서 중력·Coriolis 보상 토크로 자세 유지, TCP 축 표시

viewer에서 Ctrl + 마우스 drag로 링크를 밀면 놓은 자리에 머무름(관절 마찰 10 N·m)
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.robots.indy7_v3.constants import INDY7_V3_JOINT_NAMES, INDY7_V3_SCENE_XML  # noqa: E402
from source.sim import MujocoKernel, SimScheduler  # noqa: E402

CTRL_HZ = 500


def main():
    kernel = MujocoKernel(INDY7_V3_SCENE_XML, INDY7_V3_JOINT_NAMES)
    kernel.reset_to_keyframe()
    model, data = kernel.model, kernel.data
    tcp_id = model.site("tcp").id

    mujoco.mj_forward(model, data)
    print(f"관절 수 {len(INDY7_V3_JOINT_NAMES)}, 총 질량 {model.body_mass.sum():.2f} kg")
    print(f"home q [rad]: {np.round(kernel.joint_pos, 3)}")
    print(f"TCP 위치 [m]: {np.round(data.site_xpos[tcp_id], 4)}")

    def on_control(t):
        # 직전 mj_step의 qfrc_bias = C(q, dq)·dq + g(q)
        kernel.apply_joint_torque(data.qfrc_bias[kernel.dof_ids])

    def on_render(t, visualizer):
        visualizer.draw_axes(data.site_xpos[tcp_id], data.site_xmat[tcp_id], size=0.15)

    SimScheduler(model, data, ctrl_hz=CTRL_HZ).run(on_control, on_render)


if __name__ == "__main__":
    main()
