from openai import OpenAI

# client = OpenAI(api_key="your-api-key-here") # 如果想在代码中设置Api-key而不是全局变量就用这个代码
client = OpenAI()

with open("environment.txt", 'r') as env_file:
    Highway = env_file.read()
with open("vehicle_descriptions.txt", 'r') as mot_file:
    motion1 = mot_file.read()
with open("vehicle_descriptions_2.txt", 'r') as mot_file2:
    motion2 = mot_file2.read()  
messages = [
    {"role": "system", "content": "You are an expert in autonomous driving, reinforcement learning and code generation. We are going to use a simulated vehicle to complete given driving tasks."},
    {"role": "user", "content":  
f'''
Typically, the reward function of a manipulation task is consisted of these following parts (some part is optional, so only include it if really necessary):
1. progress quickly on the road;
2. avoid collisions.
3. difference between current state of the agent and its goal state
4. lane keeping, measured by the deviation from the center of the lane
5. regularization of the vehicle's actions to encourage smooth driving
The following is the environment of the agent.
{Highway}

You are allowed to use any existing python package if applicable. But only use these packages when it's really necessary.
I want it to fulfill the following task: The ego-vehicle is driving on a multilane highway populated with other vehicles. The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.
1. Please think step by step and tell me what does this task mean;
2. Then write a function that format as `def _reward(self, action) -> float` and returns the reward. Just the function body is fine.
3. Do not invent any variable or attribute that is not given.
4. When you writing code, you can also add some comments as your thought.
'''
},
    {"role": "assistant", "content": 
    '''
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
     '''
    },
    {"role": "user", "content": 
    f'''
Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can evaluate the agent and get a text file which details the motion data of an autonomous vehicle during evaluation. 
The following block is about the motion details.
{motion1}
Please read the motion data carefully, consider the possible improvement of the reward function cautiously and generate a new reward functon. 
    '''
    },
    {"role": "assistant", "content": 
'''
Analyzing the provided motion data and considering the behavior and performance of the agent, we can make some deductions and adjustments to the reward function to potentially improve the autonomous driving strategy in terms of speed optimization, collision avoidance, and lane positioning.

### Observations from Motion Data:
1. **Speed Maintenance**: The vehicle maintains a high speed relatively consistently but there are some speed drops, possibly due to traffic interactions.

2. **Collision Avoidance**: There are several instances where collisions are avoided, but the agent's speed is impacted. The current reward function needs to balance the objectives of collision avoidance while maintaining high velocity and lane discipline.

3. **Lane Keeping**: There are multiple lane changes. While the reward function encourages driving in the rightmost lane, continuous lane changes can be indicative of either overtaking maneuvers or inefficiency in lane stability.

### Proposed Reward Function Adjustments:
1. **Increase rewards for maintaining high speed**, especially when not near other cars, to incentivize the agent to optimize for higher speed wherever possible safely.
   
2. **Introduce a penalty for unnecessary lane changes** unless it is to avoid a slower vehicle. This will encourage lane stability and can reduce unnecessary maneuvers which potentially lead to safety hazards or inefficiency.

3. **Enhance collision avoidance rewards** not just based on actual collisions but also on the distance to the nearest vehicle. This can help in making the ride safer and smoother by preventing hard brakes or abrupt turns.
  
4. **Enhance lane keeping rewards** based on the time spent in the correct lane and introduce slight penalties for middle lanes unless overtaking.

### New Reward Function:
def _reward(self, action) -> float:
    reward = 0.0

    # 1. High speed reward scaled by absence of nearby vehicles
    close_vehicles = self.road.close_vehicles_to(self.vehicle, 50 ,see_behind=False)
    if not close_vehicles:
        reward += 0.5 * (self.vehicle.speed / self.config["reward_speed_range"][1])

    # 2. Collision penalty
    if self.vehicle.crashed:
        reward -= 5.0
    
    # 3. Lane keeping: More reward for staying in the rightmost lane reasonably
    if self.vehicle.lane_index[2] == self.config['lanes_count'] - 1:
        reward += 0.1
    elif self.vehicle.lane_index[2] == 0:
        reward -= 0.05  # slight penalty for being in the leftmost lane

    # 4. Penalty for lane changing to discourage unnecessary changes
    if action == 0 or action == 2:
        reward -= 0.1

    # 5. Keeping safe distance from other cars
    if close_vehicles:
        for v in close_vehicles:
            if np.linalg.norm(self.vehicle.position - v.position) < 20:
                reward -= 0.5  # increase this penalty if too close to the other vehicle
    return reward
'''
    },
    {"role": "user", "content": 
    f'''
Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can evaluate the agent and get a text file which details the motion data of an autonomous vehicle during evaluation. 
The following block is about the motion details.
{motion2}
Please read the motion data carefully, consider the possible improvement of the reward function cautiously and generate a new reward functon. 
    '''
    }
  ]
completion = client.chat.completions.create(
  model="gpt-4-turbo",
  seed=2024,
  messages=messages
)
with open('prompt_iter2_2.txt', 'a') as file:
    for i in range(len(messages)):
        file.write(messages[i]['content'])

with open('generation_iter2_2.txt', 'w') as file:
    file.write(completion.choices[0].message.content)  # 将字符串写入文件
# print(completion)
# Hello! How can I assist you today?



# while True:
#     user_input = input('User: ')
#     if user_input=='quit':
#         break
#     messages.append({"role": "user", "content": user_input})
#     completion = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
#     answer = completion.choices[0].message.content
#     print('System:', answer)
#     messages.append({"role": "system", "content": answer})

# response = client.chat.completions.create(
#     model='gpt-4-turbo',
#     messages=[{'role': 'user', 'content': "如何做一道麻婆豆腐"}],
#     temperature=1, # 设置想象力程度越大想象力越丰富
#     stream=True  # this time, we set stream=True
# )

# for chunk in response:
#     print(chunk.choices[0].delta.content,end='')

# messages = [
#     {"role": "system", "content": "请你扮演一个塔罗牌占卜师"},
# ]


# while True:
#     user_input = input('User:')
#     messages.append({"role": "user", "content": user_input})
#     response = client.chat.completions.create(model="gpt-4-turbo", messages=messages, stream=True)
#     answer = ''
#     for chunk in response:
#         token = chunk.choices[0].delta.content
#         if token != None:
#             answer += token
#             print(token, end='')

#     messages.append({"role": "assistant", "content": answer})
#     print()
    




