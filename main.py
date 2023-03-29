
import gymnasium as gym
from pendulum_model import PendulumModel
from CEM import CEMPlanner
import numpy as np
import matplotlib.pyplot as plt

def run_planning():
    planner=CEMPlanner()

    env = gym.make("Pendulum-v1", render_mode="human")
    observation, info = env.reset(seed=42)

    belief_model = PendulumModel()

    reward_buffer=[]
    for _ in range(100):
        action = planner.act_plan(belief_model,observation)
        print("std",planner.sigma.max())
        observation, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
        #     observation, info = env.reset()
        reward_buffer.append(reward)
        planner.reset()
    env.close()
    plt.plot(reward_buffer)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title("reward curve with step")
    plt.show()

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    run_planning()

