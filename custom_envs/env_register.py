from gymnasium.envs.registration import register

# register(
#     id="ContinuousCartPole-v0",
#     entry_point="custom_envs.continuous_cartpole_v4:ContinuousCartPoleEnv",
#     max_episode_steps=500,
# )

register(
    id="MountainCar-v1",
    entry_point="custom_envs.mountain_car_v1:MountainCarEnv",
    max_episode_steps=200,
)

register(
    id="CellGroup-v0",
    entry_point="custom_envs.cell_group:ProstateCancerTherapyEnv",
    max_episode_steps=10000,  # TODO change
)

register(
    id="LV2Populations-v0",
    entry_point="custom_envs.lotka_volterra_2_populations:LV2PopulationsEnv",
    max_episode_steps=5000,  # TODO change
)