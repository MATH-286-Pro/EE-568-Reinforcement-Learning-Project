import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboard.backend.event_processing import event_accumulator
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_tensor_result(log_dir):
    # 读取所有标量
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={ 'scalars': 10000 }
    )
    ea.Reload()

    # 拿到 “rollout/ep_rew_mean” 曲线
    rew = ea.Scalars('rollout/ep_rew_mean')
    steps_r = [x.step for x in rew]
    values_r = [x.value for x in rew]

    # 拿到 “rollout/ep_len_mean” 曲线
    length = ea.Scalars('rollout/ep_len_mean')
    steps_l = [x.step for x in length]
    values_l = [x.value for x in length]

    # 创建两个并排子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 子图1：reward
    ax1.plot(steps_r, values_r)
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_xlim(0, np.max(steps_r) + 1000)
    ax1.set_ylim(0, np.max(values_r) + 10)
    ax1.set_title("PPO: Mean Episode Reward")
    ax1.grid(True)

    # 子图2：episode length
    ax2.plot(steps_l, values_l)
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Mean Episode Length")
    ax2.set_xlim(0, np.max(steps_l) + 1000)
    ax2.set_ylim(0, np.max(values_l) + 10)
    ax2.set_title("PPO: Mean Episode Length")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()




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





# 原始颜色列表，用于不同目录的区分
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue',
          'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon',
          'gold', 'lightpurple', 'darkred', 'darkblue']


def lighten_color(color, amount=0.5):
    """
    将给定颜色与白色混合以获得更浅的色调。
    :param color: matplotlib 支持的颜色字符串
    :param amount: 混合比例，0-1，值越大越接近白色
    """
    rgb = np.array(mcolors.to_rgb(color))
    white = np.ones(3)
    return tuple(rgb + (white - rgb) * amount)


def load_monitor_results(path: str) -> pd.DataFrame:
    """
    读取指定目录下所有 monitor.csv 日志并合并成一个 DataFrame，按时间排序并重置 t 使最早为 0。
    """
    pattern = os.path.join(path, "*.monitor.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No monitor files found in {path}")

    dfs = []
    t_starts = []
    for fn in files:
        with open(fn, 'r') as f:
            header = json.loads(f.readline().lstrip('#'))
            t_starts.append(header.get('t_start', 0))
            df = pd.read_csv(f)
            df['t'] += header.get('t_start', 0)
            dfs.append(df)
    result = pd.concat(dfs, ignore_index=True).sort_values('t').reset_index(drop=True)
    result['t'] -= min(t_starts)
    return result


def plot_result(dirs, num_timesteps=None, xaxis='timesteps', task_name='', window=100):
    """
    简化版绘图：上下两个子图分别显示 Episode Rewards 和 Episode Length，保留原始颜色，并使点和线有细微色差。
    :param dirs: 日志目录列表
    :param num_timesteps: 最大 timesteps（按 episode 长度累计）
    :param xaxis: 'timesteps' | 'episodes' | 'walltime_hrs'
    :param task_name: 图标题前缀
    :param window: 平滑窗口大小
    """
    series = []
    for idx, folder in enumerate(dirs):
        df = load_monitor_results(folder)
        if num_timesteps is not None:
            df = df[df['l'].cumsum() <= num_timesteps]
        if xaxis == 'timesteps':
            df['x'] = df['l'].cumsum()
        elif xaxis == 'episodes':
            df['x'] = np.arange(len(df))
        elif xaxis == 'walltime_hrs':
            df['x'] = df['t'] / 3600.0
        else:
            raise ValueError(f"Unknown xaxis: {xaxis}")
        series.append((df, os.path.basename(folder), COLORS[idx % len(COLORS)]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    for df, label, color in series:
        light_color = lighten_color(color, 0.5)
        
        # Episode Rewards                 s=2
        axes[0].scatter(df['x'], df['r'], s=2, label=f'{label} raw')   #  c=light_color, alpha=0.8,
        if len(df) >= window:
            smoothed = df['r'].rolling(window).mean().iloc[window-1:] 
            axes[0].plot(df['x'].iloc[window-1:], smoothed, c=color, linewidth=1.5, label=f'{label} {window}-step avg')
        
        # Episode Length                  s=2
        axes[1].scatter(df['x'], df['l'], s=2, label=f'{label} raw')   #  c=light_color, alpha=0.8,
        if len(df) >= window:
            smoothed_l = df['l'].rolling(window).mean().iloc[window-1:]
            axes[1].plot(df['x'].iloc[window-1:], smoothed_l, c=color, linewidth=1.5, label=f'{label} {window}-step avg')

    axes[0].set_title(f'{task_name} Rewards')
    axes[0].set_ylabel('Episode Rewards')
    axes[0].grid(True)
    axes[1].set_title(f'{task_name} Episode Length')
    axes[1].set_xlabel(xaxis)
    axes[1].set_ylabel('Episode Length')
    axes[1].grid(True)

    # 图例
    for ax in axes:
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.show()