import os
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from callback import SaveOnRewardThreshold

def make_env(seed: int):
    def _init():
        env = gym.make("CartPole-v1")
        env.seed(seed)
        return env
    return _init

if __name__ == "__main__":
    # 日志目录
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("Training", current_time)
    os.makedirs(log_path, exist_ok=True)

    # 构造 8 个环境
    n_envs = 8
    env_fns = [make_env(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    env = VecMonitor(vec_env, filename=os.path.join(log_path, "vec_monitor.csv"))

    # PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
        n_steps=128,              # 128 % 8 == 0
    )

    # 回调
    save_cb = SaveOnRewardThreshold(threshold=200,
                                    save_path=os.path.join(log_path, "Intermediate"))
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=500,
                                            verbose=1)

    # 训练
    model.learn(total_timesteps=100_000, callback=[save_cb, stop_cb])
    model.save(os.path.join(log_path, "model_full_training"))
