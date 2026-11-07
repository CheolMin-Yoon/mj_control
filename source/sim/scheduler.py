"""MuJoCo passive viewer 시뮬 루프, physics·control·render 다중 주기와 real-time pacing"""

import time

import mujoco
import mujoco.viewer

from .visualizer import Visualizer


class SimScheduler:
    """physics는 model.opt.timestep, control은 ctrl_hz, render는 render_hz

    인자:
        model: MjModel
        data: MjData
        ctrl_hz: `on_control(t)` 호출 주기 [Hz]
        render_hz: viewer.sync + `on_render(t, visualizer)` 주기 [Hz]
    """

    def __init__(self, model, data, ctrl_hz, render_hz=60):
        self.model = model
        self.data = data
        self.ctrl_dt = 1.0 / ctrl_hz
        self.render_dt = 1.0 / render_hz
        self.sim_time = 0.0

    def run(self, on_control, on_render=None, duration=None, warn_overrun=0.05):
        """viewer가 닫히거나 sim 시간이 duration [s]에 닿을 때까지 실행"""
        self.data.time = 0.0
        self.sim_time = 0.0

        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=True, show_right_ui=True) as viewer:
            visualizer = Visualizer(viewer)

            # 정수 카운터 → float 누적 drift 방지
            ctrl_k = 0
            render_k = 0

            wall_start = time.perf_counter()
            sim_time_prev = 0.0
            overrun_warned = False

            while viewer.is_running():
                if duration is not None and self.sim_time >= duration:
                    break

                next_render = (render_k + 1) * self.render_dt
                while self.sim_time < next_render:
                    if self.sim_time >= ctrl_k * self.ctrl_dt:
                        on_control(self.sim_time)
                        ctrl_k += 1

                    mujoco.mj_step(self.model, self.data)
                    self.sim_time = self.data.time

                visualizer.reset()
                if on_render is not None:
                    on_render(self.sim_time, visualizer)
                viewer.sync()
                render_k += 1

                # sim 시간 정지 = viewer 일시정지 → wall_start 보정으로 재개 뒤 lag 폭주 방지
                wall_now = time.perf_counter()
                if self.sim_time == sim_time_prev:
                    wall_start = wall_now - self.sim_time
                sim_time_prev = self.sim_time

                slack = (wall_start + self.sim_time) - wall_now
                if slack > 0:
                    time.sleep(slack)
                elif not overrun_warned and slack < -warn_overrun:
                    print(f"[SimScheduler] sim lagging wall-clock by {-slack * 1000:.1f} ms")
                    overrun_warned = True
