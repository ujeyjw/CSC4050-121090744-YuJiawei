from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
import torch as th

from torch.nn import functional as F
import sys
from typing import Dict, Text
import gymnasium as gym
import random
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
custom_path = '/home/qi47/codes/project_reward/text2reward/run_highway/'
sys.path.insert(0, custom_path)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.envs.registration import register
from stable_baselines3.ppo_lag import PPOLagrangian


class HighwayEnvChangeReward(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
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
                "collision_reward": -5,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.8,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config
    # def seed(self, seed=None):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     # Ensure all environment-specific components that require seeding are handled here
    #     return [seed]
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=25
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
    
class PPOLagrangianWandbCallback(BaseCallback):
    def __init__(self):
        super(PPOLagrangianWandbCallback, self).__init__()

    def _on_step(self) -> bool:
        # Fetch the latest logged values from Logger
        logs = dict(self.logger.name_to_value)
        wandb.log(logs)  # Log these values to WandB
        return True

    def _on_training_end(self):
        # Optionally log at the end of training
        logs = dict(self.logger.name_to_value)
        wandb.log(logs)

def generate_vehicle_descriptions(agent_vehicle, vehicles, proximity_threshold=80):
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
        f"Speed: {agent_vehicle.speed:.2f} m/s, "
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
                    f"Speed: {vehicle.speed:.2f} km/h, "
                    f"Direction: {np.rad2deg(vehicle.heading):.2f} degrees, "
                    f"Collision Status: {'Collided' if vehicle.crashed else 'No collision'}, "
                    f"Distance from Agent: {distance:.2f} m"
                )

    return descriptions
def evaluate_model_and_record_video(model, video_filename='evaluation.mp4', num_episodes=5, description_filename='vehicle_descriptions_lag.txt', if_wandb=False):
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
                # breakpoint()
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
        artifact = wandb.Artifact('vehicle_descriptions_local_1', type='dataset')
        artifact.add_file(description_filename)

        # Use the log_artifact method to log the artifact to wandb
        wandb.log_artifact(artifact)

        # Log the video separately
        wandb.log({"evaluation_video": wandb.Video(video_filename, fps=5, format="mp4")})

def cost_function(obs, action) -> float:
    """
    Calculate a cost based on the velocity of the ego-vehicle, penalizing speeds under 15.
    :param obs: Observations/state of the environment, assumed to include velocity components of the ego vehicle.
    :param action: The last action taken by the agent (not used in this implementation).
    :return: Cost value as a float, with a penalty if speed is below 15.
    """
    # Constants for velocity threshold and penalty
    
    SPEED_THRESHOLD = (25)/(80)  # Minimum acceptable speed
    
    SPEED_PENALTY = 5 # Penalty applied if below the speed threshold
    
    # Extract the velocity components from the observation
    vx = obs[0][3]  # Assuming 'vx' is at index 3 of the first vehicle (ego vehicle)
    vy = obs[0][4]  # Assuming 'vy' is at index 4 of the first vehicle (ego vehicle)
    
    # Calculate the current speed from components
    current_speed = vx
    
    # Apply penalty if speed is below the threshold
    if current_speed < SPEED_THRESHOLD:
        return SPEED_PENALTY
    else:
        return 0.0  # No cost if speed is at or above the threshold

register(
        id='highway-custom-v0',
        entry_point='text2reward.run_highway.cppo:HighwayEnvChangeReward')

# def make_env(env_id, rank, seed=0):
#     def _init():
#         env = gym.make(env_id)
#         env.unwrapped.seed(seed + rank)
#         return env
#     return _init
if __name__ == "__main__":

    if_wandb = True
    
    config={
            "policy_type": "MlpPolicy",
            "total_timesteps": 300000,
            "env_name": 'highway-custom-v0'}
    if if_wandb:
        wandb.init(project="highway", entity="emanon47", config=config, name="highway-custom-lag_3")

    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        # env = SubprocVecEnv([make_env(config["env_name"], i) for i in range(n_cpu)])
        env = make_vec_env("highway-custom-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        env = gym.make(config["env_name"]) 
        model = PPOLagrangian(
            policy=config["policy_type"],
            env=env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            learning_rate=5e-4,
            verbose=2,
            tensorboard_log="./highway_ppo_lagrangian_tensorboard/",
            device='cuda'
        )
        
        # model = PPOLagrangian.load("highway_ppo_lagrangian_model_2")
        # model.set_env(env)
        wandb_callback = PPOLagrangianWandbCallback()

        model.learn(total_timesteps=config["total_timesteps"], cost_function = cost_function,callback=wandb_callback)
        model_path = "highway_ppo_lagrangian_model_3"
        model.save(model_path)
        if if_wandb:
            wandb.save(model_path)

    model = PPOLagrangian.load("highway_ppo_lagrangian_model_3")
    evaluate_model_and_record_video(model, if_wandb=if_wandb)
    if if_wandb:
        wandb.finish()
