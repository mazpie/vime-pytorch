from gym.envs.registration import registry, register, make, spec
# Algorithmic
# ----------------------------------------

register(
    id='MountainCarSparse-v0',
    entry_point='envs.mountain_car_sparse:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='HalfCheetahSparse-v3',
    entry_point='envs.half_cheetah_sparse:HalfCheetahEnv',
    max_episode_steps=500,
    reward_threshold=4800.0,
)