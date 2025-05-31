from gymnasium.envs.registration import register

register(
    id="ContinuousCartPole-v0",
    entry_point="custom_envs.continuous_cartpole_v4:ContinuousCartPoleEnv",
    max_episode_steps=500,  # or whatever suits your env
)
