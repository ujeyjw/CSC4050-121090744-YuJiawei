from openai import OpenAI

client = OpenAI(api_key="") # 如果想在代码中设置Api-key而不是全局变量就用这个代码
# client = OpenAI()

with open("environment.txt", 'r') as env_file:
    Highway = env_file.read()
with open("vehicle_descriptions.txt", 'r') as mot_file:
    motion1 = mot_file.read()
with open("vehicle_descriptions_2.txt", 'r') as mot_file2:
    motion2 = mot_file2.read()  
with open("vehicle_descriptions_3.txt", 'r') as mot_file3:
    motion3 = mot_file3.read()  
with open("vehicle_descriptions_4.txt", 'r') as mot_file4:
    motion4 = mot_file4.read()  
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

Please read the motion data carefully, consider the possible improvement of the reward function cautiously and generate a new reward functon. 
    '''
    },
        {"role": "assistant", "content": 
'''
From the motion data, it is evident that the agent has managed to avoid collisions efficiently, however, the speed of the agent vehicle remains relatively constant and somewhat conservative. The movements across lanes suggest that the agent is active in changing lanes yet it seems to maintain a middle pace not fully capitalizing on higher speed potentials which could be crucial for tasks involving efficient path completion.

### Revised Reward Function:

1. **Speed Optimization**: Incentivize maintaining high speeds when safe conditions are detected (lane is clear ahead).

2. **Lane Efficiency**: Encourage the agent to not just avoid the leftmost lane unless for overtaking but also seek the fastest lane when in clear road conditions. 

3. **Safe Distance Maintenance**: Boost the rewards for maintaining a more dynamic safe distance, adjusting based on speed and traffic density.

4. **Aggressive Lane Changing**: Implement a smarter lane change reward that evaluates the necessity of the change based on traffic speed in adjacent lanes and ahead.

5. **Handling High-speed Lane Changes**: Reward high-speed lane changes that lead directly to overtaking and improvement in travel time.

```python
def _reward(self, action) -> float:
    reward = 0

    # High speed reward optimization
    max_speed = self.config["reward_speed_range"][1]
    speed_fraction = self.vehicle.speed / max_speed
    reward += speed_fraction * 0.5  # Scale factor for speed reward

    # Collision penalty
    if self.vehicle.crashed:
        reward -= 5  # Significant penalty for collisions
        
    # Lane efficiency
    lanes_total = self.config['lanes_count']
    lane_id = self.vehicle.lane_index[2]
    if lane_id < lanes_total / 2:
        # Less reward for being in lanes closer to the median
        reward -= (lanes_total / 2 - lane_id) * 0.05
    else:
        # Bonus for being in lanes closer to the rightmost lane
        reward += (lane_id - lanes_total / 2) * 0.1

    # Aggressive lane changing based on traffic conditions
    if action == 0 or action == 2:
        reward -= 0.1  # General penalty to minimize excessive lane changing
        vehicles_ahead = self.road.neighbour_vehicles(self.vehicle)
        if vehicles_ahead[0]:
            # Increasing reward if the lane change contributes to overtaking
            if vehicles_ahead[0].speed < self.vehicle.speed:
                reward += 0.2
                
    # Safe distance reward/punishment
    front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
    if front_vehicle:
        distance = np.linalg.norm(self.vehicle.position - front_vehicle.position)
        safe_distance = self.vehicle.speed / 2  # Dynamic safe distance
        if distance < safe_distance:
            reward -= ((safe_distance - distance) / safe_distance)  # Proportionately penalize based on shortfall

    return reward
```
This modified reward function should promote more dynamic driving behavior, capitalize on higher speeds while ensuring safety through adaptive distance keeping, and encourage strategic lane changes. This approach should ultimately lead to improved overall performance in scenarios demanding efficient path completion and higher speeds.
'''
    },
{"role": "user", "content": 
f'''
Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can evaluate the agent and get a text file which details the motion data of an autonomous vehicle during evaluation. 
The following block is about the motion details.
{motion3}
Please read the motion data carefully, consider the possible improvement of the reward function cautiously and generate a new reward functon. 
we should encourage overtaking in order to move fast on the roads and keep a safe distance of a vehicle
'''
}
  ]


completion = client.chat.completions.create(
  model="gpt-4-turbo",
  seed=2024,
  messages=messages
)
with open('prompt_iter4_5.txt', 'a') as file:
    for i in range(len(messages)):
        file.write(messages[i]['content'])

with open('generation_iter4_5.txt', 'w') as file:
    file.write(completion.choices[0].message.content)  # 将字符串写入文件
    
# {"role": "assistant", "content": 
# '''
# Given the updated requirements for encouraging overtaking and maintaining a safe distance, let's redefine the reward function to address these new priorities effectively.

# ### Updated Reward Function Strategy:
# 1. **Overtaking Reward:** Incentivize overtaking slower vehicles by actively rewarding changes to the right lane if it leads to surpassing another vehicle.

# 2. **Safe Distance Maintenance:** Continue rewarding the maintenance of a safe following distance dynamically based on speed and traffic context, but adjust calculations to discourage tailgating more effectively.

# 3. **High-Speed Maintenance:** Encourage maintaining high speeds especially when the road ahead is clear and it's safe to do so.

# 4. **Lane Stability:** Introduce penalties for unnecessary lane changes that do not contribute to overtaking or speed maintenance.

# ### Revised Reward Function:
# ```python
# def _reward(self, action) -> float:
#         reward = 0

#         # 1. High speed reward, more significant when lanes are less occupied
#         max_speed = self.config["reward_speed_range"][1]
#         speed_fraction = self.vehicle.speed / max_speed
#         reward += speed_fraction * 0.5  # Scale factor for speed reward

#         # 2. Collision penalty
#         if self.vehicle.crashed:
#             reward -= 5

#         # 3. Lane changing and overtaking
#         if action == 0 or action == 2:  # LANE_LEFT or LANE_RIGHT
#             reward -= 0.1  # Discourage frequent lane changes
#             # Encourage overtaking: Check speed differential and position to the vehicle in the target lane
#             lead_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
#             if lead_vehicle and (self.vehicle.speed > lead_vehicle.speed):
#                 reward += 0.3  # Reward overtaking slower vehicles

#         # 4. Safe distance reward/punishment
#         front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
#         if front_vehicle:
#             distance = np.linalg.norm(self.vehicle.position - front_vehicle.position)
#             safe_distance = self.vehicle.speed / 2  # Dynamic safe distance
#             if distance < safe_distance:
#                 reward -= ((safe_distance - distance) / safe_distance)  # Proportionately penalize based on shortfall

#         return reward
# ```

# ### Explanation:
# - **High speed reward** is adjusted directly proportional to the fraction of speed relative to the maximum, promoting faster driving when possible.
# - **Collision penalty** remains strict to ensure safety concerns are paramount.
# - **Rewards for overtaking** are now more pronounced, encouraging the agent to pass slower vehicles by moving to faster lanes while being penalized for unnecessary lane changes.
# - **Safe distance maintenance** is dynamic, relating the required safe distance to the current speed of the agent vehicle. The reward/penalty adjusts dynamically based on how much smaller the actual distance is compared to the calculated safe distance.

# This revised function encourages strategic overtaking and punishes risky behaviors such as tailgating, which should lead to efficient yet safe driving patterns in high-speed environments.
# '''
#     },
# {"role": "user", "content": 
# f'''
# Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can evaluate the agent and get a text file which details the motion data of an autonomous vehicle during evaluation. 
# The following block is about the motion details.
# {motion4}
# Please read the motion data carefully, consider the possible improvement of the reward function cautiously and generate a new reward functon. 
# '''
# }

# print(completion)
# Hello! How can I assist you today?
#Human Feedback:
#One of the possible improvement: 
#When considering improvements, we could redefine the concept of "overtaking." It's not sufficient to assume that overtaking occurs whenever there is a lead vehicle and our vehicle's speed exceeds that of the lead vehicle. We need to take into account the distance between our vehicle and the lead vehicle, as well as whether the entire process of overtaking is completed.


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
    




