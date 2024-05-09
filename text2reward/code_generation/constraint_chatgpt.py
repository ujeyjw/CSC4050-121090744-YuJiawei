from openai import OpenAI

client = OpenAI(api_key="") # 如果想在代码中设置Api-key而不是全局变量就用这个代码
# client = OpenAI()
with open("ppo_lag.txt", 'r') as constraint_file:
    constraint = constraint_file.read()  
with open("environment.txt", 'r') as env_file:
    environment = env_file.read()  
with open("vehicle_descriptions_ppo.txt", 'r') as mot_file1:
    motion1 = mot_file1.read()  
with open("vehicle_descriptions_standard.txt", 'r') as mot_file2:
    motion2 = mot_file2.read()  
messages = [
    {"role": "system", "content": "You are an expert in autonomous driving, reinforcement learning and code generation. We are going to use a simulated vehicle to complete given driving tasks."},
    {"role": "user", "content": 
f"""
We are going to focus on the following RL task:
The ego-vehicle is driving on a multilane highway populated with other vehicles. The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

The method we are going to use is Proximal Policy Optimization algorithm (PPO) augmented with a Lagrangian (clip version) contraint. The following is the detail codes
{constraint}

The following is the environment of the agent
{environment}
You are allowed to use any existing python package if applicable. But only use these packages when it's really necessary.

The following is the reward function of the agent
def _reward(self, action: Action) -> float:
    '''
    The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    :param action: the last action performed
    :return: the corresponding reward
    '''
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
    forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
    scaled_speed = utils.lmap(
        forward_speed, self.config["reward_speed_range"], [0, 1]
    )
    return {{
        "collision_reward": float(self.vehicle.crashed),
        "right_lane_reward": lane / max(len(neighbours) - 1, 1),
        "high_speed_reward": np.clip(scaled_speed, 0, 1),
        "on_road_reward": float(self.vehicle.on_road),
    }}
Feed these reward codes into the environment, and use the RL algorithm to train the policy. After training, I can evaluate the agent and get a text file which details the motion data of an autonomous vehicle during evaluation. 
The following block is about the motion details.
{motion1}

As a comparison, I would like to show you my ideal motion data for agent in this task. Please compare these two motion data and think about what kind of cost function we should write.
{motion2}

I want it to design a constraint and write a cost function
1. Please think step by step and tell me what does this task mean;
2. Then write a function that format as `def cost_function(obs, action) -> float` and returns the cost. Just the function body is fine.
3. Do not invent any variable or attribute that is not given.
4. When you writing code, you can also add some comments as your thought.
"""}
]
completion = client.chat.completions.create(
  model="gpt-4-turbo",
  seed=2024,
  messages=messages
)
with open('prompt_lag_1.txt', 'a') as file:
    for i in range(len(messages)):
        file.write(messages[i]['content'])

with open('generation_lag_1.txt', 'w') as file:
    file.write(completion.choices[0].message.content)  # 将字符串写入文件
