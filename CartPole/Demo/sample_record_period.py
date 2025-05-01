import logging

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

training_period = 250           # record the agent's episode every 250
num_training_episodes = 10_000  # total number of training episodes

env = gym.make("CartPole-v1",              # replace with your environment
               render_mode="rgb_array")  

env = RecordVideo(env, 
                  video_folder="CartPole/cartpole-record", 
                  name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)

env = RecordEpisodeStatistics(env)

for episode_num in range(num_training_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:

        # Action 动作
        action = env.action_space.sample()  # replace with actual agent

        # Run Env 运行环境
        obs, reward, terminated, truncated, info = env.step(action)

        # Is Over? 是否结束
        episode_over = terminated or truncated

    logging.info(f"episode-{episode_num}", info["episode"])
env.close()