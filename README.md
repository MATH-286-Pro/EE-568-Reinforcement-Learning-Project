# **Instructions**


## **Project Guide**
1. go to folder `Project` and `open main.py`
2. run `main.py` will automatically create training data in `Training` folder will model.zip and relevant data


## **Gym CartPole**
- **CartPole-v1**
  - 存在问题：即使杆子倾斜，只要小于15°就仍然会收到 reward=1 的奖励 [[参考 Github Bug Report]](https://github.com/Farama-Foundation/Gymnasium/issues/790)


## 向量化分类

- **串行/同步** 
    - 同一个 agent 与 env1 交互完成后，再与 env2 交互，依次类推
    - 如果一个环境的主要行为 $\text{step}(s,a) \to s', r$ 很快，如 `CartPole-v1`，那么适合用

- **并行/异步** 
    - 不同的 agent, agent1 与 env1 交互，agent2 与 env2 交互，agent3 与 env3 交互，依次类推
    - 如果一个环境的主要行为 $\text{step}(s,a) \to s', r$ 很慢，使得 **进程切换的开销 < step(s,a) 的时间**，那么适合用SubprocEnv来并行训练

## 向量化封装函数

### **Gymnasium**   
```python
vec_env = gym.make_vec("cartPloe-v1", num_envs=4, vectorization_mode = "sync")                  # 串行
vec_env = gym.make_vec("cartPloe-v1", num_envs=4, vectorization_mode = "vector_entry_point")    # 并行
```

### **stable_baselines3** https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html  
```python
# 向量化 API
vec_env = DummyVecEnv(...)     # 串行
vec_env = SubprocVecEnv(...)   # 并行

# 简化 API，默认使用 DummyVecEnv
vec_env = make_vec_env(env_id      = "CartPole-v1",
                       n_envs      = 16             # 环境数量一般不超过 CPU 核心数    # 我的电脑是 16 核
                       vec_env_cls = DummyVecEnv    # 不写默认为 DummyVecEnv 串行
                       monitor_dir = None,          # 监控目录
                      )   
```

注意：使用 SubprocVecEnv 时必须要放在 `if __name__ == '__main__':` 下面，否则会报错  

```python
obs, rewards, dones, infos  = vec_env.step(actions) # stable_baselines3 返回4个值

obs, reward, terminated, truncated, info = env.step(action) # gymnasium 返回5个值
```

## **Gym 环境自定义**



## **RLHF 基于人类反馈的强化学习**



<!-- 2.4 串行环境DummyVecEnv(VecEnv)
如果一个环境的主要行为step(s, a)-> s', r很快，如 `CartPole-v1`，那么适合用DummyVecEnv来“并行”训练

    小总结VecEnv：数据对象有num_envs, observation_space, action_space, reward_range，核心行为是step()，处理的是单个IO到多个IO的类型转换
    DummyVecEnv：新增vectorized环境实体的数据对象self.envs = list(env0, env1,...)，step_wait()的行为逻辑是在self.envs中串行执行的

2.5 并行环境SubprocVecEnv(VecEnv)
如果一个环境的主要行为step(s, a)-> s', r很慢，使得进程切换的开销 < step(s,a)的时间，那么适合用SubprocEnv来并行训练，阅读stable_baselines3.vec_env.sub_proc_env的源码

    SubprocVecEnv：需要为每一个环境创建一个进程，然后由主进程进行管理，而不像DummyVecEnv那样用list()来存储多个环境
    主要逻辑：

    1. 默认用forkserver的方式，起一个资源管理进程ctx
    2. 建立n个环境对应的进程 e_0,... ,e_n, 在 ctx 与 e_i 之间建立管道pipe的两个连接Connect对象，进行数据交换
    3. 通过管道之间进行数据通信(send(),recv())，用step_async给 e_i 进程的环境发送命令与数据，并设置等待标识self.waiting = True
    4. 等所有子进程里的环境step完成，再发送给ctx进程，最后在ctx进程中将各环境的数据重组成 (num_envs, data)的形式 -->



<!-- ## **Learning**
[OpenAI Gym Website](https://gymnasium.farama.org/)


## **Code**
```bash
# install gymnasium    # 安装 gymnasium
pip install gymnasium

# Upgrade gymnasium    # 升级 gymnasium
pip install --upgrade gymnasium
```

## **Gym** 

```python
import gymnasium as gym

# Create the environment CartPole-v1  # 创建环境 倒立摆
env = gym.make('CartPole-v1')  

# Interact with the environment       # 与环境交互
observation, reward, ... = env.step(action)               

```


## **Stable-baselines3** -->