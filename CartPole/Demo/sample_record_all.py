import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 4

env = gym.make("CartPole-v1", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, 
                  video_folder    = "CartPole/cartpole-record", 
                  name_prefix     = "eval",
                  episode_trigger = lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:

        # Action 动作
        action = env.action_space.sample()  # replace with actual agent
        
        # Run Env 运行环境
        obs, reward, terminated, truncated, info = env.step(action)

        # Is Over? 是否结束
        episode_over = terminated or truncated

env.close()

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')