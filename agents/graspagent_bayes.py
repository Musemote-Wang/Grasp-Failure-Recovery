"""
GraspAgent_bayes
Created by Yue Wang on 2024-12-20
Version 1.1
config = {'env': 'GraspEnv_v1/v2/v3'}
Note:
env在这里只起到一个数据容器的作用, 保存候选抓取。
x_samples: 从信念分布中按照sigma规则采样H个状态点
z_samples: 计算H个状态点下, attempt次抓取尝试的观测
w_samples: 计算H个状态点对应的观测出现的概率, 并更新H个状态点的权重
"""

import graspenvs
import gymnasium
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np

class GraspAgent_bayes:
    def __init__(self, config):
        self.env = gymnasium.make(id=config['env'], render_mode='rgb_array')
        self.env.reset()
        self.H, self.beta = 100, 0.5 * 25

    def calculate_scores(self): # 计算候选抓取的分数
        candidate_actions = self.env.state['candidate_actions']
        t_vec = np.column_stack((np.cos(candidate_actions[:, 2]), np.sin(candidate_actions[:, 2])))
        convex = np.repeat(np.reshape(self.env.state['convex'], (-1, 8, 2)), candidate_actions.shape[0], axis=0)
        d_points = convex - np.reshape(candidate_actions[:, :2], (candidate_actions.shape[0], -1, 2))
        norm_points = d_points[:, :, 0] * t_vec[:, :1] + d_points[:, :, 1] * t_vec[:, 1:]
        d_com = self.mu[1:] - candidate_actions[:, :2]
        norm_com = d_com[:, :1] * t_vec[:, :1] + d_com[:, 1:] * t_vec[:, 1:]
        max_norm_point = np.max(norm_points / norm_com, axis=1) * norm_com[:, 0]
        scores = self.mu[0] * (max_norm_point - norm_com[:, 0]) / max_norm_point
        self.scores = np.clip(scores, 0, 1)
        self.env.state['scores'] = np.clip(scores, 0, 1)

        return np.clip(scores, 0, 1)
    
    def choose_action(self, num_topk=100):
        scores = self.calculate_scores()
        topk = np.argsort(scores)[-num_topk:]
        topk_probs = scores[topk] / np.sum(scores[topk])
        action = self.env.state['candidate_actions'][np.random.choice(topk, p=topk_probs)]

        return [action]
    
    def sigma_x_samples(self, H=100): # 从信念分布中按照sigma规则采样H个状态点
        filtered_x_samples = []
        while True:  # 保证sigma是正定的
            eigenvalues = np.linalg.eigvals(self.sigma)
            if np.any(eigenvalues <= 0):
                self.sigma += np.eye(3)
            else:
                break
        while len(filtered_x_samples) < H: # 采样H个状态点
            epsilons = np.random.multivariate_normal(np.zeros(3), np.eye(3), H)
            x_samples = self.mu + np.dot(1 * epsilons, np.linalg.cholesky(self.sigma).T)
            for x_sample in x_samples:
                if mplPath.Path(self.env.state['convex']).contains_point(x_sample[1:]):
                    filtered_x_samples.append(x_sample)
        filtered_x_samples = np.array(filtered_x_samples)

        return filtered_x_samples

    def calculate_z_samples(self, x_samples): # 计算H个状态点下, attempt次抓取尝试的观测
        z_samples = np.zeros((x_samples.shape[0], self.env.state['attempt']))
        for k in range(self.env.state['attempt']):
            a = self.env.state['history'][k, :3]
            t_vec = np.array([np.cos(a[2]), np.sin(a[2])])
            norm_points = np.dot(self.env.state['convex'] - a[0:2], t_vec)
            norm_points = np.repeat(np.reshape(norm_points, (-1, 8)), x_samples.shape[0], axis=0)
            t_vec = np.repeat(np.reshape(t_vec, (-1, 2)), x_samples.shape[0], axis=0)
            mass = x_samples[:, 0]
            com = x_samples[:, 1:]
            d_com = com - a[:2]
            norm_com = d_com[:, :1] * t_vec[:, :1] + d_com[:, 1:] * t_vec[:, 1:]
            max_norm_point = np.max(norm_points / norm_com, axis=1) * norm_com[:, 0]
            z_samples[:, k] = mass * (max_norm_point - norm_com[:, 0]) / max_norm_point
        
        return z_samples

    def calculate_w_samples(self, z_samples, beta=0.5 * 25): # 计算H个状态点对应的观测出现的概率, 并更新H个状态点的权重
        w_samples = np.zeros(z_samples.shape[0])
        error = np.abs(z_samples - self.env.state['history'][:self.env.state['attempt'], 3])
        beta_error = beta * error
        sum_error = np.sum(beta_error, axis=1)
        w_samples = np.exp(-sum_error)
        w_samples = w_samples / (np.sum(w_samples))
        
        return w_samples
    
    def update(self, action, force):
        self.env.state['history'][self.env.state['attempt']] = np.array([action[0], action[1], action[2], force])
        self.env.state['attempt'] += 1
        x_samples = self.sigma_x_samples(self.H)
        z_samples = self.calculate_z_samples(x_samples)
        w_samples = self.calculate_w_samples(z_samples, self.beta)
        self.mu = np.average(x_samples, weights=w_samples, axis=0)
        self.sigma = np.cov(x_samples.T, aweights=w_samples) + 0.01 * np.eye(3)
        # if there exists nan or inf in the covariance matrix, replace it with 1
        self.sigma = np.nan_to_num(self.sigma)
        for i in range(3):
            for j in range(3):
                self.sigma[i, j] = np.clip(self.sigma[i, j], 0.0001, np.random.uniform(0.1, 1000))

    def reset(self, contour, convex): # 将mu初始化为形心, contour应为未归一化的轮廓
        self.env.initialize_state(contour, convex) # contour: 100*2, convex: 8*2
        # scale the contour to 0-1
        scale = np.max((contour[:, :2].max(axis=0) - contour[:, :2].min(axis=0)))
        scale_point = contour[:, :2].min(axis=0)
        contour = (contour[:, :2] - scale_point) / scale
        convex = (convex - scale_point) / scale
        geometric_center = np.mean(contour[:, :2], axis=0) # 根据contour计算形心
        self.mu = np.array([0.5, geometric_center[0], geometric_center[1]])
        self.sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])