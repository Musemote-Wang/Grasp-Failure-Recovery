import gymnasium


gymnasium.register(
    id='GraspEnv_v1',
    entry_point='graspenvs.graspenv_v1:GraspEnv_v1'
)

gymnasium.register(
    id='GraspEnv_v2',
    entry_point='graspenvs.graspenv_v2:GraspEnv_v2'
)

gymnasium.register(
    id='GraspEnv_v3',
    entry_point='graspenvs.graspenv_v3:GraspEnv_v3'
)
