from easydict import EasyDict

env_name = 'guess_the_number'
collector_env_num = 8
n_episode = 256
evaluator_env_num = 100
num_simulations = 500
update_per_collect = 100
batch_size = 256
max_env_step = int(1e4)
max_guesses = 2
action_space_size = 3
fixed_secret = False
continuous_rewards = False
reanalyze_ratio = 0.
eps_greedy_exploration_in_collect = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

guess_muzero_config = dict(
    exp_name=
    f'data_mz_ctree/{env_name[:-14]}_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        obs_shape=max_guesses * 2,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        max_guesses=max_guesses,
        action_space_size=action_space_size,
        fixed_secret=fixed_secret,
        continuous_rewards=continuous_rewards,
    ),
    policy=dict(
        model=dict(
            observation_shape=max_guesses * 2,
            action_space_size=action_space_size,
            model_type='mlp', 
            lstm_hidden_size=32,
            latent_state_dim=32,
            self_supervised_learning_loss=False,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
        ),
        cuda=False,
        env_type='not_board_games',
        game_segment_length=max_env_step,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps 
            # according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
        use_augmentation=False,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.05,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # default is 0
        n_episode=n_episode,
        eval_freq=int(1e2),
        replay_buffer_size=int(1e4),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
guess_muzero_config = EasyDict(guess_muzero_config)
main_config = guess_muzero_config

guess_muzero_create_config = dict(
    env=dict(
        type='guess_the_number',
        import_names=['zoo.guess.envs.guess_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
guess_muzero_create_config = EasyDict(guess_muzero_create_config)
create_config = guess_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
