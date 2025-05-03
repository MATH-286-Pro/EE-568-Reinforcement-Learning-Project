import os
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def test_model(model_type, model_path, n_episodes=5):

    print(model_path)

    # 初始化环境
    # Initialize the environment
    env = gym.make("CartPole-v1", render_mode="human") 
    env = DummyVecEnv([lambda: env])  
    obs = env.reset()

    # 加载模型
    # Load the model
    if model_type == "PPO":
        model = PPO.load(model_path,
                         device = "cpu",
                         env = env)
    else:
        model = DQN.load(model_path,
                        device = "cpu",
                        env = env)


    # 开始测试
    # Start testing
    for episode in range(n_episodes):
        score = 0
        done = False
        while done == False:
            action, _ = model.predict(obs)            # 获取行为  
            obs, reward, done, _ = env.step(action)   # 环境交互
            score += reward                           # 回报计算
        print(f"Episode: {episode + 1} Score: {score}")
    env.close()