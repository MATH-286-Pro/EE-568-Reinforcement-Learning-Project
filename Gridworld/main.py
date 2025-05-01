import gymnasium as gym
from env import *
from gymnasium.wrappers import FlattenObservation


gym.make("gymnasium_env/GridWorld-v0")

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)