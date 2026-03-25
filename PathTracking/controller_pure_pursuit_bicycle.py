import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Pure Pursuit Gain
                 kp=0.1, Lfc=1):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc
        self.dt = model.dt
        self.l = model.l
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        Ld = self.kp*v + self.Lfc
        
        # TODO 4.3.1: Pure Pursuit Control for Bicycle Kinematic Model
        
        # 搜尋預瞄點 (Look-ahead point)
        # 從最近點開始往後找，直到點與車子的距離大於 Ld
        target_idx = self.current_idx
        while target_idx < len(self.path) - 1:
            dist = np.sqrt((self.path[target_idx][0] - x)**2 + (self.path[target_idx][1] - y)**2)
            if dist >= Ld:
                break
            target_idx += 1
        
        target_point = self.path[target_idx]
        
        # 計算預瞄點相對於車身的夾角 alpha
        # 先算目標點相對於世界座標的角度，再減去車子本身的 yaw
        # 注意：np.arctan2(dy, dx) 得到的是弧度
        dx = target_point[0] - x
        dy = target_point[1] - y
        target_angle_rad = np.arctan2(dy, dx)
        
        # 計算 alpha (弧度)
        alpha = target_angle_rad - np.deg2rad(yaw)
        
        # 根據公式計算下一個轉向角 delta (弧度)
        # 公式: delta = arctan(2 * L * sin(alpha) / Ld)
        # self.l 在 __init__ 中已定義為 model.l (軸距)
        delta_rad = np.arctan2(2.0 * self.l * np.sin(alpha), Ld)
        
        # 將弧度轉回角度 (degree) 返回給模擬器
        next_delta = np.rad2deg(delta_rad)

        # 限制在正負 40 度以內
        max_steer = 40.0
        next_delta = np.clip(next_delta, -max_steer, max_steer)

        # [end] TODO 4.3.1
        return next_delta
