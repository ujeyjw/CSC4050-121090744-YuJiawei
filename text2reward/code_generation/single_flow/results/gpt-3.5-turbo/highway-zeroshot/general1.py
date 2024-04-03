def _reward(self, action) -> float:
    reward = 0
    
    # Check if the vehicle has crashed
    if self.vehicle.crashed:
        reward += self.config["collision_reward"]
    
    # Calculate reward for driving on the right-most lanes
    reward += self.config["right_lane_reward"] * (self.vehicle.lane_index[2] / (self.road.network.all_side_lanes - 1))
    
    # Calculate reward for driving at high speed
    if self.vehicle.speed >= self.config["reward_speed_range"][0]:
        reward += self.config["high_speed_reward"] * (self.vehicle.speed - self.config["reward_speed_range"][0]) / (self.config["reward_speed_range"][1] - self.config["reward_speed_range"][0])
    
    return reward