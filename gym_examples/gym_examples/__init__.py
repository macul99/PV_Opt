from gymnasium.envs.registration import register

register(
    id="gym_examples/PanelGridWorld-v0",
    entry_point="gym_examples.envs:PanelGridWorldEnv",
    max_episode_steps=500,
)

register(
    id="gym_examples/PanelGridWorld-v1",
    entry_point="gym_examples.envs:PanelGridWorldEnv_V1",
    max_episode_steps=500,
)

register(
    id="gym_examples/PanelGridWorld-v2",
    entry_point="gym_examples.envs:PanelGridWorldEnv_V2",
    max_episode_steps=500,
)
