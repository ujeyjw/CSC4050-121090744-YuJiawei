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

kinematics = """

"""
environment_mapping = {
    "Kinematics" : 
    
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
