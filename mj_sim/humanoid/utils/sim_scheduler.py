import time
import mujoco
import mujoco.viewer

from .visualizer import Visualizer


class SimScheduler:
    """
    MuJoCo viewer 시뮬 루프 + multi-rate 스케줄링.
    - physics: model.opt.timestep (예: 0.002s)
    - control: ctrl_hz로 user callback 호출 (catch-up: 한 step에 여러 ctrl tick 가능)
    - render:  render_hz로 viewer.sync + overlay callback 호출
    - real-time pacing 자동, overrun 시 주기적 재경고
    """

    def __init__(self, model, data, ctrl_hz=500, render_hz=60):
        self.model, self.data = model, data
        self.ctrl_dt    = 1.0 / ctrl_hz
        self.render_dt  = 1.0 / render_hz
        self.physics_dt = model.opt.timestep
        self.sim_time   = 0.0

        # ctrl_hz가 1/physics_dt를 초과하면 catch-up도 의미 없음 (physics 한 번에 ctrl 다회 호출)
        assert self.ctrl_dt >= self.physics_dt, \
            f"ctrl_hz는 1/physics_dt={1/self.physics_dt:.0f} 이하여야 함 (현재 {1/self.ctrl_dt:.0f})"


    def run(self, on_control, on_render=None, duration=None,
            warn_overrun: float = 0.05, warn_cooldown: float = 1.0):
        """
        on_control(sim_time)            : data.ctrl 세팅하는 콜백 (idempotent 가정)
        on_render(sim_time, visualizer) : 오버레이 그리기 콜백
        duration                        : None이면 viewer 닫을 때까지
        warn_overrun                    : sim이 wall-clock보다 이만큼(초) 뒤처지면 경고
        warn_cooldown                   : 경고 후 재출력까지 최소 간격(초)
        """
        # 시뮬 상태 초기화 — keyframe[0] 있으면 그 자세, 없으면 zero state (data.time도 0)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.sim_time = 0.0

        with mujoco.viewer.launch_passive(self.model, self.data,
                                          show_left_ui=True,
                                          show_right_ui=True) as viewer:
            view       = Visualizer(viewer)
            next_ctrl  = next_render = 0.0
            wall_start = None                 # 첫 iter 끝나고 보정 (viewer setup time 제외)
            last_warn  = -float('inf')

            while viewer.is_running():
                if duration is not None and self.sim_time >= duration:
                    break

                # 한 render 주기 동안 physics + control 진행
                while self.sim_time < next_render:
                    # ctrl tick: 누적된 만큼 catch-up
                    while self.sim_time >= next_ctrl:
                        on_control(self.sim_time)
                        next_ctrl += self.ctrl_dt
                    mujoco.mj_step(self.model, self.data)
                    self.sim_time = self.data.time

                view.reset()
                if on_render is not None:
                    on_render(self.sim_time, view)
                viewer.sync()
                next_render += self.render_dt

                # 첫 frame 끝난 시점에 wall_start 정렬 (viewer launch_passive setup 제외)
                if wall_start is None:
                    wall_start = time.perf_counter() - self.sim_time

                slack = (wall_start + self.sim_time) - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -warn_overrun:
                    now = time.perf_counter()
                    if now - last_warn >= warn_cooldown:
                        print(f"[SimScheduler] sim lagging wall-clock by {-slack*1000:.1f} ms "
                              f"(ctrl_hz/render_hz too high?)")
                        last_warn = now
