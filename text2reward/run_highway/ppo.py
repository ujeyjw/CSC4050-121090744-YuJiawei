import os
import sys
import gymnasium as gym
import imageio
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import highway_env
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
                average_reward = sum(self.episode_rewards) / self.window_size
                wandb.log({"average_episode_reward": average_reward})
                self.episode_rewards = []
        return True

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
def evaluate_model_and_record_video(model, video_filename='evaluation.mp4', num_episodes=5, description_filename='vehicle_descriptions_ppo.txt', if_wandb=False):
    env = gym.make('highway-v0', render_mode='rgb_array')
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
        

def main(mode='train'):
    if_wandb = False
    config={
            "policy_type": "MlpPolicy",
            "total_timesteps": 200000,
            "env_name": 'highway-v0'}
    if if_wandb:
        wandb.init(project="highway", entity="emanon47", config=config, name="highway-v0-3")

    checkpoint_dir = './checkpoints/'
    latest_checkpoint = max([checkpoint_dir + d for d in os.listdir(checkpoint_dir)], key=os.path.getmtime, default=None)

    if mode == 'train':
        n_cpu = 6
        env = make_vec_env("highway-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(
            config.policy_type,
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=(64 * 12) // n_cpu,
            batch_size=64,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="./highway_ppo_tensorboard/",
            device='cuda'
        )
        
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=checkpoint_dir,
                                                 name_prefix='rl_model')
        reward_logger = CustomRewardLogger(window_size=20)
        model.learn(total_timesteps=config.total_timesteps, callback=[checkpoint_callback, reward_logger])
        model_path = "highway_ppo_model"
        model.save(model_path)
        wandb.save(model_path)
    elif mode == 'eval':
        if latest_checkpoint is not None:
            model = PPO.load(latest_checkpoint)
            evaluate_model_and_record_video(model)
        else:
            print("No checkpoints found for evaluation.")
    if if_wandb:
        wandb.finish()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        main(mode='eval')
    else:
        main(mode='train')
