import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel

class KinematicModelBicycle(KinematicModel):
    def __init__(self,
            l = 30,     # distance between rear and front wheel
            dt = 0.05
        ):
        # Distance from center to wheel
        self.l = l
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, cstate:ControlState) -> State:
        # TODO 2.3.1: Bicycle Kinematic Model
        # v, w, x, y, yaw = 0, 0, state.x, state.y, state.yaw
        # 1. 更新當前速度 v (基於加速度 a)
        v = state.v + cstate.a * self.dt
        
        # 2. 獲取當前轉向角 delta (需轉為弧度 Radian 進行三角函數運算)
        delta_rad = np.deg2rad(cstate.delta)
        yaw_rad = np.deg2rad(state.yaw)
        
        # 3. 計算角速度 w (即 theta 的變化率)
        # 公式: w = (v / L) * tan(delta)
        # 注意: 這裡的 w 單位會是 rad/s
        w_rad = (v / self.l) * np.tan(delta_rad)
        
        # 4. 更新位置 x, y (使用 Euler Integration)
        x = state.x + v * np.cos(yaw_rad) * self.dt
        y = state.y + v * np.sin(yaw_rad) * self.dt
        
        # 5. 更新航向角 yaw (將 rad/s 轉回 deg/s 後積分)
        w = np.rad2deg(w_rad)
        yaw = (state.yaw + w * self.dt) % 360

        # [end] TODO 2.3.1
        state_next = State(x, y, yaw, v, w)
        return state_next
