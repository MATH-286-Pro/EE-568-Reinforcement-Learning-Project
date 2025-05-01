# **Instructions**


## Learning
[OpenAI Gym Website](https://gymnasium.farama.org/)


## Gym 知识

```python
import gymnasium as gym

# 创建环境 倒立摆
env = gym.make('CartPole-v1')  

# 与环境交互
observation, reward, ... = env.step(action)               

```


## Code
```bash
# install gymnasium    # 安装 gymnasium
pip install gymnasium

# Upgrade gymnasium    # 升级 gymnasium
pip install --upgrade gymnasium
```

## Stable-baselines3