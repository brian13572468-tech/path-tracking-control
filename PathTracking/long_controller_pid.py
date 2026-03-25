import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class PIDLongController(Controller):
    def __init__(self, model, a_range, kp=1.5, ki=5, kd=0):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.dt = model.dt
        self.a_range = a_range
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
            return None, None
        
        # Extract State
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 1:
            # Brake to 0 speed using PID when finishing the track
            v_ref = 0.0
            target = self.path[-1]
            return np.clip(-v, self.a_range[0], self.a_range[1]), target
        else:
            # Search Nearest Target Locally
            min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
            self.current_idx = min_idx
            target = self.path[min_idx]
            v_ref = target[4]
        
        # TODO 3.2: PID Control for Longitudinal Motion
        # 1. 計算當前誤差
        error = v_ref - v
        
        # 2. 計算誤差積分 (Accumulated Error)
        self.acc_ep += error * self.dt
        
        # 3. 計算誤差微分 (Error Derivative)
        diff_ep = (error - self.last_ep) / self.dt
        
        # 4. PID 公式組合
        next_a = self.kp * error + self.ki * self.acc_ep + self.kd * diff_ep
        
        # 5. 更新上一次的誤差紀錄
        self.last_ep = error
        
        # 6. 物理限制 (Saturation): 確保加速度在 a_range 範圍內
        next_a = np.clip(next_a, self.a_range[0], self.a_range[1])
        # [end] TODO 3.2

        return next_a, target