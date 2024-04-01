其他依葫芦画瓢都挺简单，但是highway环境的pythonic abstraction可能需要对highway-env及其它的代码有比较好的理解
consider an abstraction of the environment

首先我得知道reward function应该包含哪些环境中的值
一个简单的方法观察不同任务的标准reward function
那么主要是config, vehicle and road
当然我要先完成一个task再谈后续的完整性
那么这个task是第一个task，即highway
![202404011428693.png](https://s2.loli.net/2024/04/01/jflK8yERYXehDug.png)
首先是config，值得注意的是，各种task的config都是基于某个abc
```python
def default_config(cls) -> dict:
        """
        Default environment configuration.
        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
	         "render_agent": True, 
	        "offscreen_rendering":os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False,
        }
......
#in highway
config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4, 
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }

        )
```

结合简化一下然后加上有些变量需要的注释

```python  
self.config = {
"observation": {"type": "Kinematics"},
"action": {
	"type": "DiscreteMetaAction",
},
"lanes_count": 4, 
"vehicles_count": 50,
"controlled_vehicles": 1, #the number of controlled vehicles
"initial_lane_id": None, #id of the lane to spawn in
"ego_spacing": 2, # ratio of spacing to the front vehicle, 1 being the default
"vehicles_density": 1, 
"collision_reward": -1,  # The reward received when colliding with a vehicle.
"right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
# zero for other lanes.
"high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
# lower speeds according to config["reward_speed_range"].
"lane_change_reward": 0,  # The reward received at each lane change action.
"reward_speed_range": [20, 30], #the speed range which can get reward
"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
"normalize_reward": True, #normalize reward to [0, 1]
}
self.road = Road(
	network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
record_history=self.config["show_trajectories"],
        )

self.vehicle = self.controlled_vehicles[0]
other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )
self.controlled_vehicles = []
for others in other_per_controlled:
	vehicle = Vehicle.create_random(
		self.road,
		speed=25,
		lane_id=self.config["initial_lane_id"],
		spacing=self.config["ego_spacing"],
	)
	vehicle = self.action_type.vehicle_class(
		self.road, vehicle.position, vehicle.heading, vehicle.speed
	)
	self.controlled_vehicles.append(vehicle)
	self.road.vehicles.append(vehicle)
	for _ in range(others):
		vehicle = other_vehicles_type.create_random(
			self.road, spacing=1 / self.config["vehicles_density"]
		)
		vehicle.randomize_behavior()
		self.road.vehicles.append(vehicle)
```

```python
class Vehicle(RoadObject):
    """
    A moving vehicle on a road, and its kinematics.
    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """
    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.0
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -40.0
    """ Minimum reachable speed [m/s] """
    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""
	self.prediction_type = predition_type
	self.action = {"steering": 0, "acceleration": 0}
    @classmethod
    def create_random(
        cls,
        road: Road,
        speed: float = None,
        lane_from: Optional[str] = None,
        lane_to: Optional[str] = None,
        lane_id: Optional[int] = None,
        spacing: float = 1,
    ) -> "Vehicle":

        """
        Create a random vehicle on the road.
        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.
        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
       
    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.
        Only the vehicle dynamics are copied, other properties are default.
        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
  
```

感觉简单的追踪各种object，然后删掉具体实现代码，添上注释，不太行。还不够抽象，更加好的应该是一个人直观的看那个场景，然后认为什么是应该算到reward的里的，然后我们环境的abstraction就有一个对应的pythonic的变量。然后让llm生成比较general的reward function，然后我们再把它转化为specific，可以用到原本env 框架的reward function
![](https://s2.loli.net/2024/04/01/jflK8yERYXehDug.png)
```python
def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }
```

但是提取obstraction的过程很有可能会损失一些信息。假如有标准的reward function做参考都还好，但是要是没有参考，可能会麻烦一点，不过这大概是为什么需要human feedback的原因之一。问题不大，因为我们就是让llm在我们给它的环境信息中求reward function.

```python
class HighwayEnv:
	self.config = {
	"observation": {"type": "Kinematics"},
	"action": {
		"type": "DiscreteMetaAction",
	},
	"lanes_count": 4, 
	"vehicles_count": 50,
	"controlled_vehicles": 1, #the number of controlled vehicles
	"initial_lane_id": None, #id of the lane to spawn in
	"ego_spacing": 2, # ratio of spacing to the front vehicle, 1 being the default
	"vehicles_density": 1, 
	"collision_reward": -1,  # The reward received when colliding with a vehicle.
	"right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
	# zero for other lanes.
	"high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
	# lower speeds according to config["reward_speed_range"].
	"lane_change_reward": 0,  # The reward received at each lane change action.
	"reward_speed_range": [20, 30], #the speed range which can get reward
	"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
	"normalize_reward": True, #normalize reward to [0, 1]
	}
	self.vehicle : Vehicle
	self.road : Road
class Vehicle:
	self.road : the road instance where the Vehicle is placed in
	self.position : cartesian position of Vehicle in the surface
	self.heading : the angle from positive direction of horizontal axis
    self.speed : cartesian speed of Vehicle in the surface
    self.lane_index : index of the lane in which the vehicle is located. the closer the vehicle is to the right, the larger the lane_index is
    self.lane : the lane in which the vehicle is located
    self.crashed : bool, whether the vehicle is crashed or not
    self.direction : np.array([np.cos(self.heading), np.sin(self.heading)])
    self.velocity : self.speed * self.direction
    self.on_road : bool, Is the object on its current lane, or off-road?
class Road:
	self.network: RoadNetwork, the road network describing the lanes
class RoadNetwork:
	def all_side_lanes:-> List[lane_index] 
	"""all lanes belonging to the same road."""
	
```
