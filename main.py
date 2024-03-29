
import gymnasium as gym
from pendulum_model import PendulumModel
from CEM import CEMPlanner
import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_reward(reward_buffer):
    plt.plot(reward_buffer)
    plt.xlabel("交互步")
    plt.ylabel("奖励")
    plt.title("单步奖励曲线")
    plt.show()

def draw_action(action_buffer):
    plt.plot(action_buffer, label='torque')
    plt.axhline(y=2, color='r', linestyle='--')
    plt.axhline(y=-2, color='r', linestyle='--')
    plt.xlabel("交互步")
    plt.ylabel("控制扭矩(N.m)")
    plt.title("控制扭矩曲线")
    plt.legend()
    plt.show()


def run_planning():
    planner=CEMPlanner()

    env = gym.make("Pendulum-v1", render_mode="human")
    observation, info = env.reset(seed=41)

    belief_model = PendulumModel()

    reward_buffer=[]
    action_buffer=[]
    action_std_buffer=[]
    for _ in range(100):
        start_time=time.time()
        action = planner.act_plan(belief_model,observation)
        # print("std",planner.sigma,"mean",planner.mu)
        observation, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
        #     observation, info = env.reset()
        reward_buffer.append(reward)
        action_buffer.append(action)
        action_std_buffer.append(planner.sigma[0])
        planner.reset()
        end_time = time.time()
        print("time",end_time-start_time)
    env.close()

    draw_reward(reward_buffer)
    draw_action(action_buffer)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    run_planning()

