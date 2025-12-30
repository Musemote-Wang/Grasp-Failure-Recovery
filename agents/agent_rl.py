# 基于stable-baselines3的RL算法，实现graspenv的智能体，训练代码如下：
#
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium
import gymnasium.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.ppo.policies import ActorCriticPolicy as PPOPolicy
import torch as th
import torch.nn as nn


class GraspAgent_rl:
    def __init__(self, config):
        self.env = gymnasium.make(id=config['env'], render_mode='rgb_array')
        self.env.reset()  # 这里的reset只是为了初始化环境，生成的contour会被下一行覆盖
        self.model = load_model(self.env, config)
    
    def choose_action(self):
        observation = self.env.get_observation()
        action, _ = self.model.predict(observation, deterministic=False)

        return action

    def update(self, action, force):
        self.env.state['history'][self.env.state['attempt']] = np.array([action[0], action[1], action[2], force])
        self.env.state['attempt'] += 1
        self.env.state['scores'] = np.zeros((self.env.state['candidate_actions'].shape[0], ))

    def reset(self, contour, convex):
        self.env.initialize_state(contour, convex)

class GraspPPOPolicy(PPOPolicy):
    def __init__(self, *args, **kwargs):
        super(GraspPPOPolicy, self).__init__(*args, **kwargs)

    def forward(self, obs: th.Tensor, candidate_actions):
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        candidate_actions = th.tensor(candidate_actions, dtype=th.float32).to(device='cuda')
        scores = th.exp(distribution.log_prob(candidate_actions - distribution.mode()))
        # can = candidate_actions.cpu().numpy()
        # sco = scores.cpu().detach().numpy()
        # plt.scatter(can[:, 0], can[:, 1], c=sco)
        # plt.savefig('scores.png')  # Modified by Yue Wang, for debugging
        topk = th.argsort(scores)[-100:]
        topk_probs = scores[topk] / th.sum(scores[topk])
        action = candidate_actions[topk[th.multinomial(topk_probs, 1)]]

        return action, values, distribution.log_prob(action)

class GraspPPO(PPO):
    def __init__(self, policy, env, **kwargs):
        super(GraspPPO, self).__init__(policy, env, **kwargs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        obs = th.tensor(observation, dtype=th.float32).unsqueeze(0).to(device='cuda')  # 添加 batch 维度
        action, _, _ = self.policy(obs, self.env.get_attr('state')[0]['candidate_actions'])

        return action.cpu().detach().numpy(), None  # 移除 batch 维度并转换为 numpy 数组
    
    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor, self.env.get_attr('state')[0]['candidate_actions'])  # Modified by Yue Wang
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

class GraspActor(Actor):
    def __init__(self, *args, **kwargs):
        super(GraspActor, self).__init__(*args, **kwargs)
    
    def forward(self, obs: th.Tensor, candidate_actions):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions, log_std = self.mu(latent_pi), self.log_std(latent_pi)
        # print(f'mean_actions: {mean_actions}, log_std: {log_std}')
        mean_actions = th.sigmoid(mean_actions)
        log_std = th.clamp(log_std, -20, 2)
        distribution = self.action_dist.proba_distribution(mean_actions, log_std)
        unscaled_candidate_actions = th.tensor(candidate_actions.copy(), dtype=th.float32).to(device='cuda')
        scaled_candidate_actions = th.tensor(candidate_actions.copy(), dtype=th.float32).to(device='cuda')
        scaled_candidate_actions[:, 2] = scaled_candidate_actions[:, 2] / (2 * np.pi)
        scores = th.exp(distribution.log_prob(scaled_candidate_actions,scaled_candidate_actions))  # Modified by Yue Wang
        scores = th.nan_to_num(scores, nan=0.01)  # Modified by Yue Wang
        scores = th.clamp(scores, 0.01, 1)  # Modified by Yue Wang
        if th.isnan(scores).any() or th.isinf(scores).any():
            print(f'mean_actions: {mean_actions}, log_std: {log_std}')
            print(f'candidate_actions: {candidate_actions}, mean_actions: {mean_actions}, log_std: {log_std}, scores: {scores}')
        # can = scaled_candidate_actions.cpu().numpy()
        # sco = scores.cpu().detach().numpy()
        # plt.scatter(can[:, 0], can[:, 1], c=sco)
        # plt.savefig('scores.png')  # Modified by Yue Wang, for debugging
        topk = th.argsort(scores)[-100:]
        topk_probs = scores[topk] / th.sum(scores[topk])
        index = th.multinomial(topk_probs, 1)
        action = unscaled_candidate_actions[topk[index]]
        log_prob = distribution.log_prob(scaled_candidate_actions[topk[index]], scaled_candidate_actions[topk[index]])
        return action, log_prob, log_prob

class GraspSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(GraspSACPolicy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None) -> GraspActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        
        return GraspActor(**actor_kwargs).to(self.device)
    
    def forward(self, obs: th.Tensor, candidate_actions):
        action, log_prob, _ = self.actor(obs, candidate_actions)
        
        return action, log_prob, log_prob

class GraspSAC(SAC):
    def __init__(self, policy, env, **kwargs):
        super(GraspSAC, self).__init__(policy, env, **kwargs)
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):  # Modified by Yue Wang
        obs = th.tensor(observation, dtype=th.float32).unsqueeze(0).to(device='cuda')  # 添加 batch 维度
        action, _, _ = self.policy(obs, self.env.get_attr('state')[0]['candidate_actions'])

        return action.cpu().detach().numpy(), None  # 移除 batch 维度并转换为 numpy 数组
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise,
        n_envs: int = 1,
    ):
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            # unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            unscaled_action = self.env.get_attr('state')[0]['candidate_actions'][np.random.randint(0, len(self.env.get_attr('state')[0]['candidate_actions']), 1)]  # Modified by Yue Wang
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            # scaled_action = self.policy.scale_action(unscaled_action)
            scaled_action = unscaled_action  # Modified by Yue Wang

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = scaled_action + action_noise()  # Modified by Yue Wang

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = scaled_action
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
    def collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer,
        action_noise = None,
        learning_starts: int = 0,
        log_interval = None,
    ):
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


def load_model(env, config):
    env_id, model_id = config['env'], config['model']
    if model_id == 'PPO':
        model = GraspPPO(GraspPPOPolicy, env, verbose=1)
    elif model_id == 'SAC':
        model = GraspSAC(GraspSACPolicy, env, verbose=1)
    model.load(f'./checkpoint/{env_id}/rl/{model_id}/model_best')
    
    return model

