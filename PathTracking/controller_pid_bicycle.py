import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.1.3: Tune PID Gains
                 kp=0.052175, 
                 ki=0.000986, 
                 kd=0.248264):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.dt = model.dt
        self.current_idx = 0
    
    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0
        self.current_idx = 0
    
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State
        x, y, yaw = info["x"], info["y"], info["yaw"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        # TODO 4.1.3: PID Control for Bicycle Kinematic Model

        
        # 取得最近的目標點
        target = self.path[min_idx]
        
        # 1. 計算目標點相對於當前位置的絕對角度 (deg)
        # target[0] 是 x, target[1] 是 y
        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0] - x))
        
        # 2. 計算角度偏差 (Target Angle - Current Yaw)
        theta_err = theta_target - yaw
        
        # 3. 核心 Error Definition: 橫向偏差 (Cross-Track Error)
        # 利用 sin 將直線距離投影到車身垂直方向
        err = min_dist * np.sin(np.deg2rad(theta_err))
        
        # --- PID 計算部分 (先寫好結構) ---
        self.acc_ep += err * self.dt
        diff_ep = (err - self.last_ep) / self.dt
        
        # PID 輸出：這裡的輸出是前輪轉向角 delta
        next_delta = self.kp * err + self.ki * self.acc_ep + self.kd * diff_ep

        # 假設前輪最大轉向角為 40 度
        max_delta = 40.0 
        next_delta = np.clip(next_delta, -max_delta, max_delta)
        
        self.last_ep = err

        # [end] TODO 4.1.3
        return next_delta
