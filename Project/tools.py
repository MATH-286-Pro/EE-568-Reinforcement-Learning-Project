import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd

def plot_result(log_dir):
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


################################################################
## Loading
################################################################
class LoadMonitorResultsError(Exception):
    """当没有找到任何 monitor 文件时抛出"""
    pass

class Monitor:
    # SB3 和 gym 中 monitor wrapper 默认的文件后缀
    EXT = ".monitor.csv"

def get_monitor_files(path: str) -> list[str]:
    """
    在指定目录下查找所有以 Monitor.EXT 结尾的文件

    :param path: 目录路径
    :return: 符合条件的文件路径列表
    """
    pattern = os.path.join(path, f"*{Monitor.EXT}")
    return glob.glob(pattern)

def load_results(path: str) -> pd.DataFrame:
    """
    Load all Monitor logs from a given directory path matching `*monitor.csv`

    :param path: the directory path containing the log file(s)
    :return: the logged data as a single pandas DataFrame
    """
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")

    data_frames = []
    headers = []
    for file_name in monitor_files:
        with open(file_name, "r") as f:
            # 第一行是以 '#' 开头的 JSON header
            first_line = f.readline()
            if not first_line.startswith("#"):
                raise LoadMonitorResultsError(f"Unexpected header format in {file_name}")
            header = json.loads(first_line[1:])
            headers.append(header)

            # 剩余部分读作 CSV
            df = pd.read_csv(f, index_col=None)
            # gym Monitor 的 t 列是相对于该文件开始的时间，这里平移到全局
            df["t"] = df["t"] + header.get("t_start", 0)
        data_frames.append(df)

    # 合并所有文件的数据
    result = pd.concat(data_frames, ignore_index=True)
    # 按时间排序
    result.sort_values("t", inplace=True)
    result.reset_index(drop=True, inplace=True)

    # 将 t 列平移，使最早的 t=0
    min_t0 = min(h.get("t_start", 0) for h in headers)
    result["t"] = result["t"] - min_t0

    return result
    
################################################################
## Plotting
################################################################
X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_EPLEN = True
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


def rolling_window(array, window):
    """
    apply a rolling window to a np.ndarray

    :param array: (np.ndarray) the input Array
    :param window: (int) length of the rolling window
    :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1, var_2, window, func):
    """
    apply a function to the rolling window of 2 arrays

    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on variable 2 (such as np.mean)
    :return: (np.ndarray, np.ndarray)  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2


def ts2xy(timesteps, xaxis,yaxis=None):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    if yaxis is Y_EPLEN:
        y_var = timesteps.l.values
    return x_var, y_var


def plot_curves(xy_list, xaxis, title):
    """
    plot the curves

    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """

    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()

def plot_results_test(dirs, 
                 num_timesteps, 
                 xaxis, 
                 task_name):
    """
    plot the results

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int or None) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)

    #plt.figure(1)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name+'Rewards')
    plt.ylabel("Episode Rewards")
    plt.grid()
    
    #plt.figure(2)
    xy_list = [ts2xy(timesteps_item, xaxis, Y_EPLEN) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name+'Ep Len')
    plt.ylabel("Episode Length")
    plt.grid()
