"""
generate_dataset.py
Created by Yue Wang on 2024-09-10
Version 1.1
dataset = {'attempts': [], 'observations': [], 'actions': [], 'next_observations': [], 'rewards': [], 'forces': [], 'dones': [], 'com': []}
Note:
优化了代码框架, 数据集生成与agent解耦, 只依赖于env
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import graspenvs
import graspenvs.utils
import gymnasium
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np


def generate_dataset(episodes=30000):
    # 创建环境实例
    env = gymnasium.make(id='GraspEnv_v3', render_mode='rgb_array')
    # 初始化数据集
    dataset = {'attempts': [], 'observations': [], 'actions': [], 'next_observations': [], 'rewards': [], 'forces': [], 'dones': [], 'com': []}
    success, sample_flag = episodes, 0
    # 运行10000个episode
    for _ in range(episodes):
        contour, convex = graspenvs.utils.generate_contour()
        env.initialize_state(contour, convex)
        observation, info = env.reset()
        while True:
            # 随机选择动作
            if env.state['candidate_actions'].shape[0] == 0:
                break
            action = env.state['candidate_actions'][np.random.randint(0, env.state['candidate_actions'].shape[0])]
            next_observation, reward, done, truncated, info = env.step(action)
            sample_flag += 1
            if sample_flag == 15:
                sample_flag = 0
                if np.random.rand() < ((env.state['attempt'] + 7) / 23):
                    dataset['attempts'].append(env.state['attempt'])
                    dataset['observations'].append(observation)
                    dataset['actions'].append(action)
                    dataset['next_observations'].append(next_observation)
                    dataset['forces'].append(env.state['history'][env.state['attempt'] - 1, -1])
                    dataset['rewards'].append(reward)
                    dataset['dones'].append((done and not truncated))
                    dataset['com'].append(info['com'])
                
            if done and not truncated and np.random.rand() < ((env.state['attempt'] + 3) / 19):
                dataset['attempts'].append(env.state['attempt'])
                dataset['observations'].append(observation)
                dataset['actions'].append(action)
                dataset['next_observations'].append(next_observation)
                dataset['forces'].append(env.state['history'][env.state['attempt'] - 1, -1])
                dataset['rewards'].append(reward)
                dataset['dones'].append((done and not truncated))
                dataset['com'].append(info['com'])
            observation = next_observation.copy()
            if done:
                success -= truncated
                break
        if _ % 100 == 0:
            print(f"Processed {_}/{episodes} episodes")
    print(f"成功率: {100 * success / episodes:.2f}%")

    # 将数据集转换为 numpy 数组
    dataset['attempts'] = np.array(dataset['attempts'])
    dataset['observations'] = np.array(dataset['observations'])
    dataset['actions'] = np.array(dataset['actions'])
    dataset['next_observations'] = np.array(dataset['next_observations'])
    dataset['forces'] = np.array(dataset['forces'])
    dataset['rewards'] = np.array(dataset['rewards'])
    dataset['dones'] = np.array(dataset['dones'])
    dataset['com'] = np.array(dataset['com'])

    # 保存数据集到文件
    np.savez('./dataset/offline_dataset/eval_dataset_v3.npz', **dataset)
    print(f"{dataset['attempts'].shape[0]} data saved to ./dataset/offline_dataset/eval_dataset_v3.npz")

if __name__ == '__main__':
    generate_dataset()