import os, argparse

from code_generation.single_flow.zero_shot.generation_highway import ZeroShotGenerator
from code_generation.single_flow.classlike_prompt.high_way_prompt import HIGHWAY_PROMPT

instruction_mapping = {
    "Highway": "The ego-vehicle is driving on a multilane highway populated with other vehicles. The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.",
    "Merge": "The ego-vehicle starts on a main highway but soon approaches a road junction with incoming vehicles on the access ramp. The agent's objective is now to maintain a high speed while making room for the vehicles so that they can safely merge in the traffic.",
    "Roundabout": "The ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.",
    "Parking": "A goal-conditioned continuous control task in which the ego-vehicle must park in a given space with the appropriate heading.",
    "Intersection": "An intersection negotiation task with dense traffic.",
    "Racetrack": "A continuous control environment, where the he agent has to follow the tracks while avoiding collisions with other vehicles.",
}

Highway = """
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
	"ego_spacing": 2, # ratio of spacing to the front vehicle, 1 being the default
	"vehicles_density": 1, 
	"collision_reward": -1,  # The reward received when colliding with a vehicle.
	"right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
	# zero for other lanes.
	"high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
	# lower speeds according to config["reward_speed_range"].
	"lane_change_reward": 0,  # The reward received at each lane change action.
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
	#all lanes belonging to the same road.
"""
environment_mapping = {
    "Highway" : Highway
    
}

# mapping_dicts = {
#     "self.robot.ee_position": "obs[:3]",
#     "self.robot.gripper_openness": "obs[3]",
#     "self.obj1.position": "obs[4:7]",
#     "self.obj1.quaternion": "obs[7:11]",
#     "self.obj2.position": "obs[11:14]",
#     "self.obj2.quaternion": "obs[14:18]",
#     "self.goal_position": "self.env._get_pos_goal()",
# }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--TASK', type=str, default="Highway", \
                        help="choose one task from: Highway, Merge, Roundabout, Parking, Intersection, Racetrack")
    parser.add_argument('--FILE_PATH', type=str, default=None)
    parser.add_argument('--MODEL_NAME', type=str, default="gpt-4")

    args = parser.parse_args()
    # File path to save result
    if args.FILE_PATH == None:
        args.FILE_PATH = "results/{}/highway-zeroshot/{}.txt".format(args.MODEL_NAME, args.TASK)
    os.makedirs(args.FILE_PATH, exist_ok=True)

    code_generator = ZeroShotGenerator(HIGHWAY_PROMPT, args.MODEL_NAME)
    general_code, specific_code = code_generator.generate_code(instruction_mapping[args.TASK], mapping_dicts)

    with open(os.path.join(args.FILE_PATH, "general.py"), "w") as f:
        f.write(general_code)

    with open(os.path.join(args.FILE_PATH, "specific.py"), "w") as f:
        f.write(specific_code)
