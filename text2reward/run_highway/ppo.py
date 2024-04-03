# import os
# import gymnasium as gym
# import highway_env
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import EvalCallback
# import wandb
# from wandb.integration.sb3 import WandbCallback

# # 初始化wandb
# wandb.init(project="highway", entity="emanon47", 
#            config={
#                "policy_type": "MlpPolicy",
#                "total_timesteps": 20000,
#                "env_name": "highway-v0",
#            })

# config = wandb.config  # 使用wandb的配置系统

# # 创建环境
# env = make_vec_env("highway-v0", n_envs=4)

# # PPO模型设置
# model = PPO(config.policy_type, env, verbose=1, tensorboard_log=f"./highway_ppo_tensorboard/",
#             learning_rate=1e-3, n_steps=2048, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
#             ent_coef=0.0, policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]))

# # 设置评估回调
# eval_env = DummyVecEnv([lambda: gym.make("highway-v0")])
# eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
#                              log_path='./logs/', eval_freq=5000, 
#                              deterministic=True, render=False)

# # 使用wandb回调
# wandb_callback = WandbCallback(model_save_path="./logs/", verbose=2)

# # 训练模型
# model.learn(total_timesteps=config.total_timesteps, callback=[wandb_callback, eval_callback])

# # 保存模型
# model_path = os.path.join("logs", "ppo_highway")
# model.save(model_path)
# wandb.save(model_path)

# # 关闭wandb的运行
# wandb.finish()
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env  # noqa: F401
import wandb
from wandb.integration.sb3 import WandbCallback

# wandb初始化
wandb.init(project="highway_v0", entity="emanon47")

# 配置wandb
config = wandb.config
config.batch_size = 64
config.n_cpu = 6
config.total_timesteps = int(2e4)

if __name__ == "__main__":
    train = False
    if train:
        env = make_vec_env("highway-v0", n_envs=config.n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=config.batch_size * 12 // config.n_cpu,
            batch_size=config.batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="./highway_ppo/",
            device='auto', 
        )
        # Train the agent
        model.learn(total_timesteps=config.total_timesteps, callback=WandbCallback(verbose=2))
        # Save the agent
        model.save("highway_ppo/model")

    model = PPO.load("highway_ppo/model")
    env = gym.make("highway-v0", render_mode="rgb_array")
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    plt.imshow(env.render())
    plt.show()
