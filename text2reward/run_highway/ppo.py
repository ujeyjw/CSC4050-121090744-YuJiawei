import os
import highway_env
import imageio
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from matplotlib import pyplot as plt
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
    env = gym.make('highway-v0', render_mode='rgb_array')
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
if __name__ == "__main__":
    wandb.init(project="highway", entity="emanon47", config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 2e5,
        "env_name": "highway-v0"}, name="highway2",)

    config = wandb.config
    train = False
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env("highway-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
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
            tensorboard_log="./highway_ppo_tensorboard/",
        )
        # Train the agent
        model.learn(total_timesteps=config.total_timesteps, callback=CustomRewardLogger())
        # Save the agent
        model_path = "highway_ppo_model"
        model.save(model_path)
        wandb.save(model_path)
    
    model = PPO.load("highway_ppo_model")
    evaluate_model_and_record_video(model)

    wandb.finish()
