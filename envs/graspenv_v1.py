"""
GraspEnv_v1
Created by Yue Wang on 2024-08-10
Version 1.1
动作空间为1维, 历史为15*2维, 视觉为2维, 观测为2+15*2=32维
self.state = {'contour': contour, 'convex': convex, 'candidate_actions': candidate_actions, 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 2))}
Note:
直接采样候选抓取
"""

import cv2
import numpy as np
import gymnasium
from gymnasium import spaces

class GraspEnv_v1(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, render_mode='human'):
        super(GraspEnv_v1, self).__init__()
        self.max_steps = 15
        self.render_mode = render_mode
        self.state_space = {'contour': spaces.Box(low=0, high=1, shape=(2, 1)), 'convex': spaces.Box(low=0, high=1, shape=(2, 1)), 'candidate_actions': spaces.Box(low=-10, high=10, shape=(1, )), 'mass': spaces.Box(low=0, high=1, shape=(1, )), 'com': spaces.Box(low=0, high=1, shape=(1, )), 'attempt': spaces.Discrete(1), 'history': spaces.Box(low=-10, high=10, shape=(self.max_steps, 2))}
        self.observation_space = spaces.Box(low=0, high=1, shape=(32, ))
        self.action_space = spaces.Box(low=0, high=1, shape=(1, ))
        self.reset()
    
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
        if action > self.state['com'][0]:
            force = self.state['mass'][0] * self.state['com'][0] / action[0] + noise
        else:
            force = self.state['mass'][0] * (1 - self.state['com'][0]) / (1 - action[0]) + noise

        return force
    
    def compute_reward(self, force):
        # calculate the reward
        return np.abs(force / self.state['mass'][0]) * np.abs(force / self.state['mass'][0]) + 1 * (np.abs(force - self.state['mass'][0]) < 0.05) - 1 - 0 * (force < 0.1)
    
    def step(self, action):
        force = self.compute_force(action)
        self.state['history'][self.state['attempt']] = np.array([action[0], force])
        self.state['attempt'] += 1

        return self.get_observation(), self.compute_reward(force), self.is_done(force), self.is_truncated(force), self.get_info()
    
    def reset(self, seed=None):
        # initialize the env.state
        contour, convex = np.array([[0], [1]]), np.array([[0], [1]])
        mass = np.array([np.random.uniform(0, 1)])
        com = np.random.uniform(0, 1, 1)
        self.state = {'contour': contour, 'convex': convex, 'candidate_actions': np.arange(0, 1, 1 / 50), 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 2))}

        return self.get_observation(), self.get_info()
    
    def render(self):
        frame = np.ones((500, 500, 3), dtype=np.uint8) * 255
        # draw the contour
        contour = self.state['contour']
        for point in contour:
            cv2.circle(frame, (int(500 * point[0]), int(250 * 1)), 3, (0, 255, 0), -1)  # Black point
        # draw the com
        com = self.state['com']
        mass = self.state['mass']
        cv2.circle(frame, (int(500 * com[0]), int(250 * 1)), 3, (0, 0, 255), -1)  # Red point
        cv2.putText(frame, f'{25 * mass[0]:.1f}', (int(500 * com[0]) + 5, int(250 * 1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, f"Attempts: {self.state['attempt']}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # draw the history
        for i in range(self.state['attempt']):
            action_x, force = self.state['history'][i]
            action_y = 1
            if i == self.state['attempt'] - 1:
                cv2.circle(frame, (int(500 * action_x), int(250 * action_y)), 3, (255, 0, 0), -1)  # If the last action, red point
            else:
                cv2.circle(frame, (int(500 * action_x), int(250 * action_y)), 3, (0, 0, 0), -1)  # Black point
            cv2.putText(frame, f'{25 * force:.1f}', (int(500 * action_x) + 5, int(250 * action_y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        # show the frame
        if self.render_mode == 'human':
            cv2.imshow('Environment State', frame)
            cv2.waitKey(500)  # wait for 1ms
            return frame
        elif self.render_mode == 'rgb_array':
            return frame

if __name__ == '__main__':   
    pass
