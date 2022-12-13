import yaml
import numpy as np

from pendulum_model import PendulumModel


class PlannerCfg:
    def __init__(self):
        with open('config/planner.yaml') as f:
            planner_cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.action_dim = planner_cfg['action_dim']
        self.horizon = planner_cfg['horizon']
        self.max_iter = planner_cfg['max_iter']
        self.random_sample = planner_cfg['random_sample']
        self.elite_sample = planner_cfg['elite_sample']


class CEMPlanner:
    def __init__(self):
        self.cfg = PlannerCfg()
        self.mu = np.zeros((self.cfg.horizon, self.cfg.action_dim), dtype=np.float32)
        self.sigma = np.ones((self.cfg.horizon, self.cfg.action_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.cfg.random_sample, self.cfg.horizon, self.cfg.action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(self.cfg.random_sample, dtype=np.float32)

    def sample_action(self):
        self.action_buffer = np.random.normal(self.mu, self.sigma,
                                              size=(self.cfg.random_sample, self.cfg.horizon, self.cfg.action_dim))

    def evaluate_action(self, env: PendulumModel, observation0):
        for traj in range(self.cfg.random_sample):
            total_reward = 0.0
            env.reset(observation0)
            for tick in range(self.cfg.horizon):
                observation, reward, _, _, _ = env.step(self.action_buffer[traj, tick])
                total_reward += reward
            self.reward_buffer[traj] = total_reward

    def update_distribution(self, elite_index):
        select_action = self.action_buffer[elite_index]
        self.mu = np.mean(select_action, axis=0)
        self.sigma = np.std(select_action, axis=0)

    def act_plan(self,env: PendulumModel, observation_now):
        for i in range(self.cfg.max_iter):
            self.sample_action()
            self.evaluate_action(env, observation_now)
            elite_index = np.argsort(self.reward_buffer)[::-1][:self.cfg.elite_sample]
            self.update_distribution(elite_index)
        return self.mu[0]

    def reset(self):
        self.mu = np.zeros((self.cfg.horizon, self.cfg.action_dim), dtype=np.float32)
        self.sigma = np.ones((self.cfg.horizon, self.cfg.action_dim), dtype=np.float32)
