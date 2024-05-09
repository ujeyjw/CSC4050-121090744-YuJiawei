# This file is here just to define the TwoCriticsPolicy for PPO-Lagrangian
from stable_baselines3.common.policies import ActorTwoCriticCnnPolicy, ActorTwoCriticPolicy, MultiInputActorTwoCriticPolicy

TwoCriticsMlpPolicy = ActorTwoCriticPolicy
TwoCriticsCnnPolicy = ActorTwoCriticCnnPolicy
TwoCriticsMultiInputPolicy = MultiInputActorTwoCriticPolicy
