import os
import sys
import gymnasium as gym
import imageio
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

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

def evaluate_model_and_record_video(model, video_filename='evaluation.mp4', num_episodes=5):
    env = gym.make('highway-v0', render_mode='rgb_array')
    with imageio.get_writer(video_filename, fps=10) as video:
        for _ in range(num_episodes):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                frame = env.render()
                video.append_data(frame)
    wandb.log({"evaluation_video": wandb.Video(video_filename, fps=10, format="gif")})

def main(mode='train'):
    wandb.init(project="highway", entity="emanon47", config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 4e5,
        "env_name": "highway-v0"
    }, name="highway-v0-3")

    config = wandb.config
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

    wandb.finish()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        main(mode='eval')
    else:
        main(mode='train')
