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
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
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

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # rewards = self._rewards(action)
        # reward = sum(
        #     self.config.get(name, 0) * reward for name, reward in rewards.items()
        # )
        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(
        #         reward,
        #         [
        #             self.config["collision_reward"],
        #             self.config["high_speed_reward"] + self.config["right_lane_reward"],
        #         ],
        #         [0, 1],
        #     )
        # reward *= rewards["on_road_reward"]
        reward = 0
    
    # Check if the vehicle has crashed
        if self.vehicle.crashed:
            reward += self.config["collision_reward"]
        
        # Calculate reward for driving on the right-most lanes
        reward += self.config["right_lane_reward"] * (self.vehicle.lane_index[2] / (len(self.road.network.all_side_lanes(self.vehicle.lane_index)) - 1))
        
        # Calculate reward for driving at high speed
        if self.vehicle.speed >= self.config["reward_speed_range"][0]:
            reward += self.config["high_speed_reward"] * (self.vehicle.speed - self.config["reward_speed_range"][0]) / (self.config["reward_speed_range"][1] - self.config["reward_speed_range"][0])
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
def evaluate_model_and_record_video(model, video_filename='evaluation.mp4', num_episodes=5):
    env = gym.make('highway-custom-v0', render_mode='rgb_array')
    with imageio.get_writer(video_filename, fps=20) as video:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
                frame = env.render()
                video.append_data(frame)
    wandb.log({"evaluation_video": wandb.Video(video_filename, fps=20, format="gif")})
#export PYTHONPATH="${PYTHONPATH}:/home/qi47/codes/project_reward/"
register(
    id='highway-custom-v0',
    entry_point='text2reward.run_highway.myenv:HighwayEnvChangeReward')
if __name__ == "__main__":

    wandb.init(project="highway", entity="emanon47", config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 2e5,
        "env_name": 'highway-custom-v0'}, name="highway-custom-v02",)

    config = wandb.config
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env('highway-custom-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(
            config.policy_type,
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="./HighwayCustom_ppo_tensorboard/",
        )
        # Train the agent
        model.learn(total_timesteps=config.total_timesteps, callback=CustomRewardLogger())
        # Save the agent
        model_path = "HighwayCustom_ppo_model"
        model.save(model_path)
        wandb.save(model_path)
        
    model = PPO.load("HighwayCustom_ppo_model")
    evaluate_model_and_record_video(model)
    wandb.finish()




