from gym.envs.registration import register

register(
    id='wobble-v0',
    entry_point='gym_wobble.atari:WobbleEnv',
)

register(
    id='WobbleNoFrameskip-v4',
    entry_point='gym_wobble.atari:WobbleEnv',
    kwargs={'max_timesteps': 100}, # A frameskip of 1 means we get every frame
    max_episode_steps=10000,
    nondeterministic=False,
)

register(
    id='WobbleNoFrameskip-v3',
    entry_point='gym_wobble.atari:WobbleEnv',
    kwargs={'max_timesteps': 100, 'tcp_tagging': True}, # A frameskip of 1 means we get every frame
    max_episode_steps=10000,
    nondeterministic=False,
)
