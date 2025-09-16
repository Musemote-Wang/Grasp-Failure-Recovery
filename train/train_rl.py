# 基于stable-baselines3的RL算法，实现graspenv的智能体，训练代码如下：
#
import argparse
import graspenvs
import gymnasium
import numpy as np
import os
import torch
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import graspagents

# 定义学习率调度函数
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func
def exponential_schedule(initial_value, decay_rate=0.99):
    def func(progress_remaining):
        return initial_value * (decay_rate ** (1 - progress_remaining))
    return func
def cosine_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))
    return func

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='GraspEnv_v3', help='environment name')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--algorithm', type=str, default='SAC', help='algorithm type')
args = parser.parse_args()
log_dir = f'./checkpoint/{args.env}/rl/{args.algorithm}/lr_{args.lr}_bs_{args.batch_size}'
os.makedirs(log_dir, exist_ok=True)
new_logger = configure(log_dir, ["stdout", "csv"])

train_env = gymnasium.make(id=args.env, render_mode='rgb_array')
train_env = DummyVecEnv([lambda: train_env])
eval_env = gymnasium.make(id=args.env, render_mode='rgb_array')
eval_env = Monitor(eval_env, log_dir)
eval_env = DummyVecEnv([lambda: eval_env])
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=100,
                             n_eval_episodes=10, deterministic=True, render=False)

# model = graspagents.GraspSAC(
#     graspagents.GraspSACPolicy, 
#     train_env, 
#     verbose=1, 
#     learning_rate=exponential_schedule(args.lr), 
#     batch_size=args.batch_size, 
#     buffer_size=30000, 
#     learning_starts=30000, 
#     action_noise = NormalActionNoise(mean=np.zeros(3), sigma=0.01 * np.ones(3)), policy_kwargs=dict(net_arch=[128, 128, 128]))
model = SAC(
    'MlpPolicy', 
    train_env, 
    verbose=1, 
    learning_rate=exponential_schedule(args.lr), 
    batch_size=args.batch_size, 
    buffer_size=30000, 
    learning_starts=30000, 
    action_noise = NormalActionNoise(mean=np.zeros(3), sigma=0.01 * np.ones(3)), 
    policy_kwargs=dict(net_arch=[64, 128, 64]))
# model = graspagents.GraspPPO(graspagents.GraspPPOPolicy, train_env, verbose=1)
# model = DQN('MlpPolicy', train_env, verbose=1, learning_rate=exponential_schedule(args.lr), batch_size=args.batch_size, buffer_size=30000, learning_starts=3000, policy_kwargs=dict(net_arch=[128, 128, 128]))
model.set_logger(new_logger)
model.learn(total_timesteps=200000, callback=eval_callback)
model.save(os.path.join(log_dir, "model_final"))


