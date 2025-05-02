# **Instructions**


## Learning
[OpenAI Gym Website](https://gymnasium.farama.org/)


## Code
```bash
# install gymnasium    # 安装 gymnasium
pip install gymnasium

# Upgrade gymnasium    # 升级 gymnasium
pip install --upgrade gymnasium
```

## Gym 

```python
import gymnasium as gym

# Create the environment CartPole-v1  # 创建环境 倒立摆
env = gym.make('CartPole-v1')  

# Interact with the environment       # 与环境交互
observation, reward, ... = env.step(action)               

```


## Stable-baselines3