"""
MuJoCo viewer 시뮬 루프 + multi-rate 스케줄링.
- physics: model.opt.timestep
- control: ctrl_hz로 on_control(t) 호출
- render: render_hz로 viewer.sync + on_render(t, viewer) 호출
- real-time pacing + 일시정지 자동 보정 + 카운터 drift 방지
"""


import time
import mujoco
import mujoco.viewer

from .visualizer import Visualizer


class SimScheduler:
    def __init__(self, model, data, ctrl_hz=500, render_hz=60):
        self.model = model
        self.data  = data
        self.ctrl_dt   = 1.0 / ctrl_hz
        self.render_dt = 1.0 / render_hz
        self.physics_dt = model.opt.timestep
        self.sim_time = 0.0

    def run(self, on_control, on_render=None, duration=None, warn_overrun: float = 0.05):
        self.data.time = 0.0
        self.sim_time = 0.0

        with mujoco.viewer.launch_passive(self.model, self.data,
                                          show_left_ui=True,
                                          show_right_ui=True) as viewer:
            # contact 시각화는 user_scn(Visualizer)에서 별도 그림 — 내장 flag off
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

            view = Visualizer(viewer)

            # 정수 카운터 — float 누적 drift 방지
            ctrl_k   = 0
            render_k = 0

            wall_start = time.perf_counter()
            sim_time_prev = 0.0
            overrun_warned = False

            while viewer.is_running():
                if duration is not None and self.sim_time >= duration:
                    break

                # 다음 render time까지 physics + control 진행
                next_render = (render_k + 1) * self.render_dt
                while self.sim_time < next_render:
                    next_ctrl = ctrl_k * self.ctrl_dt
                    if self.sim_time >= next_ctrl:
                        on_control(self.sim_time)
                        ctrl_k += 1

                    mujoco.mj_step(self.model, self.data)
                    self.sim_time = self.data.time

                # render
                view.reset()
                if on_render is not None:
                    on_render(self.sim_time, view)
                viewer.sync()
                render_k += 1

                # 일시정지 감지 — sim_time 안 변하면 viewer 일시정지 상태
                # → wall_start를 보정해서 unpause 후 lag 폭주 방지
                wall_now = time.perf_counter()
                if self.sim_time == sim_time_prev:
                    wall_start = wall_now - self.sim_time
                sim_time_prev = self.sim_time

                # real-time pacing
                slack = (wall_start + self.sim_time) - wall_now
                if slack > 0:
                    time.sleep(slack)
                elif (not overrun_warned) and slack < -warn_overrun:
                    print(f"[SimScheduler] sim lagging wall-clock by {-slack*1000:.1f} ms")
                    overrun_warned = True
