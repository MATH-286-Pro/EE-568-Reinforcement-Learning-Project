import gym
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_       # 或直接绑定到内建 bool


env = gym.make('CartPole-v1', render_mode='human')
env.reset()
for _ in range(200):
    env.step(env.action_space.sample()) # take a random action
env.close()