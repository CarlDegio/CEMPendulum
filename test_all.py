

def test_pendulum_model():
    import gymnasium as gym
    import numpy as np
    from pendulum_model import PendulumModel
    env = gym.make('Pendulum-v1')
    observation, info=env.reset()
    belief_model=PendulumModel()
    belief_observation, belief_info=belief_model.reset(observation)
    assert np.allclose(observation,belief_observation)
    assert np.allclose(belief_model.state,env.env.env.state)
    for i in range(250):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        belief_observation, belief_reward, _, _, _=belief_model.step(action)
        assert np.allclose(observation, belief_observation, atol=1e-3)
        assert np.allclose(reward, belief_reward, atol=1e-3)

class TestCEM:
    def test_planner_cfg(self):
        from CEM import PlannerCfg
        planner_cfg=PlannerCfg()
        assert planner_cfg.action_dim==1

    def test_cem_planner_select_action(self):
        from CEM import CEMPlanner
        planner=CEMPlanner()
        planner.sample_action()
        assert planner.action_buffer.shape==(planner.cfg.random_sample, planner.cfg.horizon, planner.cfg.action_dim)

    def test_cem_planner_eval_action(self):
        from CEM import CEMPlanner
        from pendulum_model import PendulumModel
        import numpy as np
        planner=CEMPlanner()
        planner.sample_action()
        env=PendulumModel()
        observation0, info=env.reset(np.array([1.0,0.0,0.0],dtype=np.float32))
        planner.evaluate_action(env,observation0)
        assert planner.reward_buffer.shape==(planner.cfg.random_sample,)

    def test_cem_planner_update_distribution(self):
        from CEM import CEMPlanner
        from pendulum_model import PendulumModel
        import numpy as np
        planner=CEMPlanner()
        planner.sample_action()
        env=PendulumModel()
        observation0, info=env.reset(np.array([1.0,0.0,0.0],dtype=np.float32))
        planner.evaluate_action(env,observation0)
        # elite_index=np.argsort(planner.reward_buffer)[::-1][:planner.cfg.elite_sample]
        elite_index=[0,1]
        mu_shape=planner.mu.shape
        sigma_shape=planner.sigma.shape
        planner.update_distribution(elite_index)
        assert planner.mu.shape==mu_shape
        assert planner.sigma.shape==sigma_shape
