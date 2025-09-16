"""
GraspEnv_v3
Created by Yue Wang on 2024-10-10
Version 3.1
动作空间为3维, 历史为15*4维, 视觉为8*2维, 观测为8*2+15*4=76维
self.state = {'contour': contour, 'convex': convex, 'candidate_actions': candidate_actions, 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 4))}
Note: 
采用反点法计算候选抓取
物体边界增加法向量, 维度为N*3
"""

import cv2
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
import gymnasium
from gymnasium import spaces
import graspenvs.utils as utils

class GraspEnv_v3(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, render_mode='human'):
        super(GraspEnv_v3, self).__init__()
        self.max_steps = 10
        self.render_mode = render_mode
        self.state_space = {'contour': spaces.Box(low=0, high=1, shape=(100, 3)), 'convex': spaces.Box(low=0, high=1, shape=(8, 2)), 'candidate_actions': spaces.Box(low=0, high=1, shape=(3, )), 'mass': spaces.Box(low=0, high=1, shape=(1, )), 'com': spaces.Box(low=0, high=1, shape=(2, )), 'attempt': spaces.Discrete(1), 'history': spaces.Box(low=0, high=1, shape=(self.max_steps, 4))}
        self.observation_space = spaces.Box(low=0, high=1, shape=(56, ))
        self.action_space = spaces.Box(low=0, high=1, shape=(3, ))
    
    def get_observation(self):
        return np.concatenate((self.state['convex'].reshape(-1), self.state['history'].reshape(-1)))
    
    def get_info(self):
        return {'contour': self.state['contour'], 'convex': self.state['convex'], 'mass': self.state['mass'], 'com': self.state['com']}

    def is_done(self, force):
        return ((np.abs(force / self.state['mass'][0]) > 0.9) or (self.state['attempt'] >= self.max_steps - 1))

    def is_truncated(self, force):
        return ((self.state['attempt'] >= self.max_steps - 1) and (np.abs(force / self.state['mass'][0]) < 0.9))

    def compute_force(self, action):
        # calculate the force
        noise = np.random.normal(0, 0.0001)
        t_vec = np.array([np.cos(action[2]), np.sin(action[2])])
        norm_points = np.dot(self.state['contour'][:, 0:2] - action[0:2], t_vec)
        norm_com = np.dot(self.state['com'] - action[0:2], t_vec)
        max_norm_point = np.max(norm_points / norm_com) * norm_com
        force = self.state['mass'][0] * (max_norm_point - norm_com) / max_norm_point + noise

        return force
    
    def compute_reward(self, force):
        # calculate the reward
        return np.abs(force / self.state['mass'][0]) * np.abs(force / self.state['mass'][0]) + 1 * (np.abs(force / self.state['mass'][0]) > 0.9) - 1 - 0 * (force < 0.1)
    
    def step(self, action):
        # # 计算action与candidate_actions的距离
        # distances = np.linalg.norm(self.state['candidate_actions'][:, 0:2] - action[0:2], axis=1)
        # index = np.argmin(distances)  # 选择最近的候选动作
        # action = self.state['candidate_actions'][index]
        force = self.compute_force(action)
        self.state['history'][self.state['attempt']] = np.array([action[0], action[1], action[2], force])
        self.state['attempt'] += 1

        return self.get_observation(), self.compute_reward(force), self.is_done(force), self.is_truncated(force), self.get_info()
    
    def reset(self, seed=None, options=None): # initialize the contour 
        while True:
            contour, convex = utils.generate_contour()
            self.initialize_state(contour, convex)
            if self.state['candidate_actions'].shape[0] > 0:
                break
            
        return self.get_observation(), self.get_info()
    
    def render(self):
        frame = np.ones((500, 500, 3), dtype=np.uint8) * 255
        # draw the contour
        for point in self.state['contour']:
            cv2.circle(frame, (int(500 * point[0]), int(500 * point[1])), 3, (0, 255, 0), -1)  # Green point
            cv2.line(frame, (int(500 * point[0]), int(500 * point[1])), (int(500 * point[0] + 10 * np.cos(point[2])), int(500 * point[1] + 10 * np.sin(point[2]))), (0, 255, 0), 1)
        # draw the candidate actions
        cmap = plt.get_cmap('coolwarm')
        for action, score in zip(self.state['candidate_actions'], self.state['scores']):
            color = cmap(int(255 * score))[:3]
            color = tuple(int(255 * c) for c in color)
            cv2.circle(frame, (int(500 * action[0]), int(500 * action[1])), int(5 * score), color, -1)
            # cv2.line(frame, (int(500 * action[0]), int(500 * action[1])), (int(500 * action[0] - 30 * np.sin(action[2])), int(500 * action[1] + 30 * np.cos(action[2]))), (0, 0, 0), 1)
        # draw the convex
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(500 * self.state['contour'][:, 0:2], dtype=np.int32)], (0, 0, 0))
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # draw the com
        com = self.state['com']
        mass = self.state['mass']
        cv2.circle(frame, (int(500 * com[0]), int(500 * com[1])), 3, (0, 0, 255), -1)  # Red point
        cv2.putText(frame, f'{25 * mass[0]:.1f}', (int(500 * com[0]) + 5, int(500 * com[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # draw the history
        cv2.putText(frame, f"Attempts: {self.state['attempt']}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        for i in range(self.state['attempt']):
            action_x, action_y, action_theta, force = self.state['history'][i]
            if i == self.state['attempt'] - 1:
                cv2.circle(frame, (int(500 * action_x), int(500 * action_y)), 3, (255, 0, 0), -1)  # If the last action, red point
                cv2.line(frame, 
                (int(500 * action_x + 70 * np.sin(action_theta)), int(500 * action_y - 70 * np.cos(action_theta))), 
                (int(500 * action_x - 70 * np.sin(action_theta)), int(500 * action_y + 70 * np.cos(action_theta))), 
                (0, 0, 0), 1)
            else:
                cv2.circle(frame, (int(500 * action_x), int(500 * action_y)), 3, (0, 0, 0), -1)  # Black point
                cv2.line(frame, 
                (int(500 * action_x + 50 * np.sin(action_theta)), int(500 * action_y - 50 * np.cos(action_theta))), 
                (int(500 * action_x - 50 * np.sin(action_theta)), int(500 * action_y + 50 * np.cos(action_theta))), 
                (0, 0, 0), 1)
            cv2.putText(frame, f'{25 * force:.1f}', (int(500 * action_x) + 5, int(500 * action_y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        if self.render_mode == 'human':
            cv2.imshow('Environment State', frame)
            cv2.waitKey(0)  # wait for keyboard
            return frame
        elif self.render_mode == 'rgb_array':
            return frame
        
    def initialize_state(self, contour, convex): # initialize the state
        # scale the contour to 0-1
        scale = np.max((contour[:, :2].max(axis=0) - contour[:, :2].min(axis=0)))
        scale_point = contour[:, :2].min(axis=0)
        contour = (contour[:, :2] - scale_point) / scale
        convex = (convex - scale_point) / scale
        # calculate the candidate actions
        contour = utils.interpolate_contour(contour)
        contour = utils.add_normal_to_contour(contour)
        candidate_actions = calculate_candidate_actions(contour, scale)
        # initialize the mass and com
        mass = np.array([np.random.uniform(5, 25)]) / 25
        while True:
            com = np.random.uniform(0, 50, 2) / 50
            if mplPath.Path(convex).contains_point(com) and np.linalg.norm(contour[:, 0:2] - com, axis=1).min() > 0.03:
                break
        self.state = {'contour': contour, 'convex': convex, 'candidate_actions': candidate_actions, 'scores': np.zeros(candidate_actions.shape[0]), 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 4))}


def calculate_candidate_actions(contour, scale):
    # interpolate the contour with normal vectors
    # calculate the candidate actions
    p1, p2 = contour[:, np.newaxis, :], contour[np.newaxis, :, :]
    dp = p1 - p2
    distance_scores = np.linalg.norm(dp[:, :, :2], axis=2)
    angles_p1, angles_p2 = np.arctan2(dp[:, :, 1], dp[:, :, 0]), np.arctan2(-dp[:, :, 1], -dp[:, :, 0])
    cos_p1, cos_p2 = np.cos(angles_p1 - p1[:, :, 2]), np.cos(angles_p2 - p2[:, :, 2])
    antipodal_scores = np.minimum(cos_p1, cos_p2)

    x, y, theta = (p1[:, :, 0] + p2[:, :, 0]) / 2, (p1[:, :, 1] + p2[:, :, 1]) / 2, angles_p1 + np.pi / 2
    mask = (antipodal_scores > 0.9) & (distance_scores * scale < 0.3)
    candidate_actions = np.column_stack([x[mask].reshape(-1), y[mask].reshape(-1), theta[mask].reshape(-1)])

    return candidate_actions