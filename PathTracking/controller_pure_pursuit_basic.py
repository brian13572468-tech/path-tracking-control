import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBasic(Controller):
    def __init__(self, model, 
                 # Optional TODO: Tune Pure Pursuit Gain
                 kp=0.1, Lfc=1):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        Ld = self.kp*v + self.Lfc

        # Optional TODO: Pure Pursuit Control for Basic Kinematic Model
        # You can implement this if you want to use Pure Pursuit for basic kinematic model in F1 Challenge
        # 2. 尋找預瞄點 (Look-ahead point)
        # 我們從當前最近點開始往後找，找到第一個距離超過 Ld 的點
        search_idx = self.current_idx
        for i in range(self.current_idx, len(self.path)):
            dist = np.hypot(x - self.path[i, 0], y - self.path[i, 1])
            if dist >= Ld:
                search_idx = i
                break
        
        target_point = self.path[search_idx]

        # 3. 計算目標角度 alpha (弧度)
        # target_point[0]: x, target_point[1]: y
        dx = target_point[0] - x
        dy = target_point[1] - y
        
        # 目標點相對於世界座標的角度
        target_yaw = np.arctan2(dy, dx)
        
        # alpha 是目標角度與當前 yaw 的差值
        # 注意：此處 yaw 通常是角度(degree)，需轉為弧度
        alpha = utils.angle_norm(target_yaw - np.deg2rad(yaw))

        # 4. 計算角速度 next_w (degree/s)
        # 公式: omega = 2 * v * sin(alpha) / Ld
        # 為了穩定性，當 v 很小時，omega 也應該很小
        if Ld != 0:
            omega_rad = (2.0 * v * np.sin(alpha)) / Ld
            next_w = np.rad2deg(omega_rad)
        else:
            next_w = 0.0
        
        return next_w
