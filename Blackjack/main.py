import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import BlackjackAgent


#####################################################################################################
# hyperparameters 超参数
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

# Build environment 创建环境 
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Build agent 创建智能体
agent = BlackjackAgent(
    env=env,
    learning_rate   = learning_rate,
    initial_epsilon = start_epsilon,
    epsilon_decay   = epsilon_decay,
    final_epsilon   = final_epsilon,
)

#####################################################################################################
# Train the agent 训练智能体
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:

        # get the action from the agent 获取智能体的动作
        action = agent.get_action(obs)

        # Iteract with the environment 与环境交互
        next_obs, reward, terminated, truncated, info = env.step(action)  

        # update the agent 更新智能体
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()


#####################################################################################################
# Plotting the results 绘制结果
def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500 episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()