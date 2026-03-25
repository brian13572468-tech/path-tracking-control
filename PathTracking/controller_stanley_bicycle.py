import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Stanley Gain
                 kp=0.5):
        self.path = None
        self.kp = kp
        self.l = model.l
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Front Wheel Target Locally
        front_x = x + self.l*np.cos(np.deg2rad(yaw))
        front_y = y + self.l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta)) if np.cos(np.deg2rad(delta)) != 0 else v
        
        min_idx, min_dist = utils.search_nearest_local(self.path, (front_x,front_y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx]

        # TODO 4.3.1: Stanley Control for Bicycle Kinematic Model
        
        # 1. 取得目標路徑點的資訊
        # target[0]:x, target[1]:y, target[2]:yaw (path heading), target[4]:v_ref
        target = self.path[min_idx]
        theta_path = target[2]  # 路徑的切線角度 (degree)
        
        # 2. 計算航向誤差 theta_e (Heading Error)
        # 需正規化到 -180 ~ 180 度之間
        theta_e = theta_path - yaw
        while theta_e > 180: theta_e -= 360
        while theta_e < -180: theta_e += 360
        theta_e_rad = np.deg2rad(theta_e)
        
        # 3. 計算橫向誤差 e 的正負號 (Sign of cross-track error)
        # 利用向量叉積 (Cross Product) 判斷路徑點在車子左邊還是右邊
        # 車頭方向向量 (vec_yaw) 與 車中心到目標點向量 (vec_target)
        dx = target[0] - front_x
        dy = target[1] - front_y
        
        # 使用正弦定理的邏輯判斷誤差符號
        # 如果 sin(yaw_to_target - yaw) > 0，代表目標點在左側，e 應為正
        yaw_to_target = np.arctan2(dy, dx)
        angle_diff = yaw_to_target - np.deg2rad(yaw)
        # 正規化 angle_diff
        while angle_diff > np.pi: angle_diff -= 2*np.pi
        while angle_diff < -np.pi: angle_diff += 2*np.pi
        
        # 根據 Hint 2，如果點在左側 e > 0，右側 e < 0
        error_front_axle = min_dist if angle_diff > 0 else -min_dist
        
        # 4. 計算 Stanley 控制公式
        # delta = theta_e + arctan(k * e / vf)
        # 為避免低速時分母為零，加上一個小的常數 (如 0.001)
        k = self.kp
        delta_e_rad = np.arctan2(k * error_front_axle, vf + 0.001)
        
        delta_rad = theta_e_rad + delta_e_rad
        next_delta = np.rad2deg(delta_rad)
        
        # 5. 加入轉向限制 (±40度)
        max_steer = 40.0
        next_delta = np.clip(next_delta, -max_steer, max_steer)

        # [end] TODO 4.3.1
    
        return next_delta
