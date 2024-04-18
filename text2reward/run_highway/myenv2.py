from typing import Dict, Text
import gymnasium as gym
import numpy as np
import highway_env
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import os
import highway_env
import imageio
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

Observation = np.ndarray
from gymnasium.envs.registration import register



class HighwayEnvChangeReward(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        # config.update(
        #     {
        #         "observation": {"type": "Kinematics"},
        #         "action": {
        #             "type": "DiscreteMetaAction",
        #         },
        #         "lanes_count": 4,
        #         "vehicles_count": 50,
        #         "controlled_vehicles": 1,
        #         "initial_lane_id": None,
        #         "duration": 40,  # [s]
        #         "ego_spacing": 2,
        #         "vehicles_density": 1,
        #         "collision_reward": -1,  # The reward received when colliding with a vehicle.
        #         "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
        #         # zero for other lanes.
        #         "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
        #         # lower speeds according to config["reward_speed_range"].
        #         "lane_change_reward": 0,  # The reward received at each lane change action.
        #         "reward_speed_range": [20, 30],
        #         "normalize_reward": True,
        #         "offroad_terminal": False,
        #     }
        # )
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
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -10,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.2,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.5,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": -0.05,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )

        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
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

    # def _reward(self, action: Action) -> float:
    #     """
    #     The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    #     :param action: the last action performed
    #     :return: the corresponding reward
    #     """
    #     reward = self.config["collision_reward"] if self.vehicle.crashed else 0.0

    #     # Reward for driving at high speed
    #     high_speed_reward = ((self.vehicle.speed - self.config["reward_speed_range"][0]) /
    #                         (self.config["reward_speed_range"][1] - self.config["reward_speed_range"][0]))
    #     high_speed_reward = self.config["high_speed_reward"] * np.clip(high_speed_reward, 0, 1)
    #     reward += high_speed_reward
    #     print(self.road.network.all_side_lanes(self.vehicle.lane_index))
    #     # Reward for driving on the right-most lanes
    #     right_lane_reward = (self.vehicle.lane_index[2] / (len(self.road.network.all_side_lanes(self.vehicle.lane_index)) - 1))
    #     right_lane_reward = self.config["right_lane_reward"] * np.clip(right_lane_reward, 0, 1)
    #     reward += right_lane_reward

    #     # Penalty for changing lanes, to encourage smooth driving
    #     if action == 0 or action == 2:
    #         reward += self.config["lane_change_reward"]

    #     # Normalize the reward if required
    #     if self.config["normalize_reward"]:
    #         reward = np.tanh(reward)
    #     return reward
    def _reward(self, action) -> float:
        
        """
        Enhanced reward function to improve lane discipline, maintain safer distances,
        encourage smoother speed control, and discourage unnecessary lane changes.
        """
        if self.vehicle.crashed:
            return self.config["collision_reward"]  # Return the collision penalty
        
        # Calculate the speed reward: normalize the vehicle's speed to be between 0 and 1 in the desired range
        speed_reward = (
            (self.vehicle.speed - self.config["reward_speed_range"][0]) /
            (self.config["reward_speed_range"][1] - self.config["reward_speed_range"][0])
        )
        speed_reward = max(0, min(1, speed_reward)) * self.config["high_speed_reward"]  # Bound the reward between 0 and 1
        
        # Calculate the lane position reward: encourage staying in right-most lanes
        max_lane_index = max(lane[2] for lane in self.road.network.all_side_lanes(self.vehicle.lane_index))
        lane_position_reward = (
            (max_lane_index - self.vehicle.lane_index[2]) / max_lane_index
        ) * self.config["right_lane_reward"]
        
        # Optional: Add a lane change penalty or reward based on action taken
        lane_change_penalty = 0
        if action == 0 or action == 2:  # Assuming actions 0 and 2 are LANE_LEFT and LANE_RIGHT
            lane_change_penalty = self.config["lane_change_reward"]
        
        # Sum the components to get the total reward
        total_reward = speed_reward + lane_position_reward + lane_change_penalty
        
        # Optionally normalize the reward to be between 0 and 1
        if self.config.get("normalize_reward", False):
            total_reward = (total_reward + 1) / 2  # assuming rewards are between -1 and 1 initially
        
        return total_reward

    # def _rewards(self, action: Action) -> Dict[Text, float]:
    #     neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #     lane = (
    #         self.vehicle.target_lane_index[2]
    #         if isinstance(self.vehicle, ControlledVehicle)
    #         else self.vehicle.lane_index[2]
    #     )
    #     # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
    #     forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
    #     scaled_speed = utils.lmap(
    #         forward_speed, self.config["reward_speed_range"], [0, 1]
    #     )
    #     return {
    #         "collision_reward": float(self.vehicle.crashed),
    #         "right_lane_reward": lane / max(len(neighbours) - 1, 1),
    #         "high_speed_reward": np.clip(scaled_speed, 0, 1),
    #         "on_road_reward": float(self.vehicle.on_road),
    #     }
    def nearest_vehicle_distance(self) -> float:
        """
        Calculate the minimum distance to the nearest vehicle ahead in the same lane.
        
        Returns:
        float: The distance in meters to the nearest vehicle ahead on the same lane. Returns a large number if no vehicle is ahead.
        """
        min_distance = float('inf')  # Initialize with a very large number
        current_position = self.vehicle.position[0]  # Assuming position is a tuple (x, y), and x is the longitudinal position
        
        for vehicle in self.road.vehicles:  # Assuming we can iterate over all vehicles on the road
            if vehicle.lane_index == self.vehicle.lane_index and vehicle.position[0] > current_position:
                # Calculate longitudinal distance to vehicles ahead in the same lane
                distance = vehicle.position[0] - current_position
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance if min_distance != float('inf') else 0  # Return 0 if no vehicle is ahead
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
    
class CustomRewardLogger(BaseCallback):
    def __init__(self, verbose=0, window_size=20):
        super(CustomRewardLogger, self).__init__(verbose)
        self.window_size = window_size  
        self.episode_rewards = [] 
        self.current_rewards = []  

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.current_rewards.append(reward) 
        
        info = self.locals['infos'][0]
        if info.get('terminal_observation') is not None:
            total_reward = sum(self.current_rewards)
            self.episode_rewards.append(total_reward)
            self.current_rewards = []  

            if len(self.episode_rewards) >= self.window_size:
                average_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                wandb.log({"average_episode_reward": average_reward})
                self.episode_rewards = []  

        return True

# def generate_vehicle_descriptions(vehicles):
#     descriptions = []
#     for idx, vehicle in enumerate(vehicles):
#         collision_status = "Collided" if vehicle.crashed else "No collision"
#         description = (
#             f"Vehicle Index: {idx + 1}, "  # Using index as a placeholder for ID
#             f"Lane: {vehicle.lane_index[2] + 1}, "
#             f"Position: ({vehicle.position[0]:.2f}, {vehicle.position[1]:.2f}), "
#             f"Speed: {vehicle.speed:.2f} km/h, "
#             f"Direction: {np.rad2deg(vehicle.heading):.2f} degrees, "
#             f"Collision Status: {collision_status}"
#         )
#         descriptions.append(description)
#     return descriptions
def generate_vehicle_descriptions(agent_vehicle, vehicles, proximity_threshold=50):
    """ Generate descriptions only for the agent and nearby vehicles.
    
    :param agent_vehicle: The agent vehicle to focus on.
    :param vehicles: List of all vehicles.
    :param proximity_threshold: Distance threshold to consider a vehicle "nearby" (in meters).
    :return: List of descriptions for the agent and nearby vehicles.
    """
    descriptions = []
    agent_position = np.array(agent_vehicle.position)

    # Add description for the agent vehicle
    descriptions.append(
        f"Agent Vehicle -- Lane: {agent_vehicle.lane_index[2] + 1}, "
        f"Position: ({agent_vehicle.position[0]:.2f}, {agent_vehicle.position[1]:.2f}), "
        f"Speed: {agent_vehicle.speed:.2f} km/h, "
        f"Direction: {np.rad2deg(agent_vehicle.heading):.2f} degrees, "
        f"Collision Status: {'Collided' if agent_vehicle.crashed else 'No collision'}"
    )

    # Check other vehicles if they are in proximity of the agent
    for vehicle in vehicles:
        if vehicle is not agent_vehicle:
            vehicle_position = np.array(vehicle.position)
            distance = np.linalg.norm(vehicle_position - agent_position)
            if distance <= proximity_threshold:
                descriptions.append(
                    f"Nearby Vehicle -- Lane: {vehicle.lane_index[2] + 1}, "
                    f"Position: ({vehicle.position[0]:.2f}, {vehicle.position[1]:.2f}), "
                    f"Speed: {vehicle.speed:.2f} m/s, "
                    f"Direction: {np.rad2deg(vehicle.heading):.2f} degrees, "
                    f"Collision Status: {'Collided' if vehicle.crashed else 'No collision'}, "
                    f"Distance from Agent: {distance:.2f} m"
                )

    return descriptions
def evaluate_model_and_record_video(model, video_filename='evaluation.mp4', num_episodes=5, description_filename='vehicle_descriptions_local2.txt', if_wandb=False):
    env = gym.make('highway-custom-v0', render_mode='rgb_array')
    with imageio.get_writer(video_filename, fps=5) as video, open(description_filename, 'w') as desc_file:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = truncated = False
            desc_file.write(f"Episode {episode+1}:\n")
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                frame = env.render()
                video.append_data(frame)
                
                # Generate and record vehicle descriptions

                # breakpoint()
                vehicles = env.unwrapped.road.vehicles
                agent_vehicle = env.unwrapped.road.vehicles[0]
                descriptions = generate_vehicle_descriptions(agent_vehicle, vehicles)
                # descriptions = generate_vehicle_descriptions(vehicles)
                desc_file.write(f"Time: {env.unwrapped.time:.2f}s\n")
                for desc in descriptions:
                    desc_file.write(desc + '\n')
                desc_file.write('\n')
            desc_file.write('\n')  # Add a space between episodes for clarity
    if if_wandb:
        # Create a wandb Artifact for the description file
        artifact = wandb.Artifact('vehicle_descriptions', type='dataset')
        artifact.add_file(description_filename)

        # Use the log_artifact method to log the artifact to wandb
        wandb.log_artifact(artifact)

        # Log the video separately
        wandb.log({"evaluation_video": wandb.Video(video_filename, fps=5, format="mp4")})
# def evaluate_model_and_record_video(model, video_filename='evaluation.mp4', num_episodes=5, description_filename='vehicle_descriptions.txt', if_wandb=False):
#     env = gym.make('highway-custom-v0', render_mode='rgb_array')
#     with imageio.get_writer(video_filename, fps=5) as video, open(description_filename, 'w') as desc_file:
#         for episode in range(num_episodes):
#             obs = env.reset()
#             done = truncated = False
#             desc_file.write(f"Episode {episode+1}:\n")
#             while not done and not truncated:
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, reward, done, truncated, info = env.step(action)
#                 frame = env.render()
#                 video.append_data(frame)

#                 # Assuming the first vehicle is the agent
#                 agent_vehicle = env.unwrapped.road.vehicles[0]
#                 vehicles = env.unwrapped.road.vehicles
#                 descriptions = generate_vehicle_descriptions(agent_vehicle, vehicles)
#                 desc_file.write(f"Time: {env.unwrapped.time:.2f}s\n")
#                 for desc in descriptions:
#                     desc_file.write(desc + '\n')
#                 desc_file.write('\n')
#             desc_file.write('\n')
#     if if_wandb:
#         # Create a wandb Artifact for the description file
#         artifact = wandb.Artifact('vehicle_descriptions', type='dataset')
#         artifact.add_file(description_filename)

#         # Use the log_artifact method to log the artifact to wandb
#         wandb.log_artifact(artifact)

#         # Log the video separately
#         wandb.log({"evaluation_video": wandb.Video(video_filename, fps=5, format="mp4")})

register(
    id='highway-custom-v0',
    entry_point='text2reward.run_highway.myenv2:HighwayEnvChangeReward')

if __name__ == "__main__":
    if_wandb = False
    
    config={
            "policy_type": "MlpPolicy",
            "total_timesteps": 200000,
            "env_name": 'highway-custom-v0'}
    if if_wandb:
        wandb.init(project="highway", entity="emanon47", config=config, name="highway-custom-v2_4.0",notes="trajectory guided")

    train = False
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env('highway-custom-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(
            config["policy_type"],
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=(64 * 12) // n_cpu,
            batch_size=64,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="./HighwayCustom_ppo_tensorboard/",
            device = 'cuda'
        )
        model.learn(total_timesteps=config["total_timesteps"], callback=CustomRewardLogger())
        model_path = "HighwayCustom_ppo_model_4_trajectory"
        model.save(model_path)
        if if_wandb:
            wandb.save(model_path)

    model = PPO.load("HighwayCustom_ppo_model_4_trajectory")
    evaluate_model_and_record_video(model, if_wandb=if_wandb)
    if if_wandb:
        wandb.finish()



