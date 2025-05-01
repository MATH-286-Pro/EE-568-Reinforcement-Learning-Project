import gymnasium as gym
env = gym.make('CartPole-v1')  # 创建环境 倒立摆

action = None

env.step(action)