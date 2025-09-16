"""
GraspEnv_v2
Created by Yue Wang on 2024-09-10
Version 2.1
动作空间为3维, 历史为15*4维, 视觉为8*2维, 观测为8*2+15*4=76维
self.state = {'contour': contour, 'convex': convex, 'candidate_actions': candidate_actions, 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 4))}
Note: 
采用矩形拟合法计算候选抓取
轮廓采用等比例缩放法
物体边界增加法向量, 维度为N*3
"""

import cv2
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
import gymnasium
from gymnasium import spaces
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster, linkage
import graspenvs.utils as utils

class GraspEnv_v2(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, render_mode='human'):
        super(GraspEnv_v2, self).__init__()
        self.max_steps = 10
        self.render_mode = render_mode
        self.state_space = {'contour': spaces.Box(low=0, high=1, shape=(100, 3)), 'convex': spaces.Box(low=0, high=1, shape=(8, 2)), 'candidate_actions': spaces.Box(low=-10, high=10, shape=(3, )), 'mass': spaces.Box(low=0, high=1, shape=(1, )), 'com': spaces.Box(low=0, high=1, shape=(2, )), 'attempt': spaces.Discrete(1), 'history': spaces.Box(low=-10, high=10, shape=(self.max_steps, 4))}
        self.observation_space = spaces.Box(low=0, high=1, shape=(76, ))
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
        norm_points = np.dot(self.state['contour'][:, :2] - action[0:2], t_vec)
        norm_com = np.dot(self.state['com'] - action[0:2], t_vec)
        max_norm_point = np.max(norm_points / norm_com) * norm_com
        force = self.state['mass'][0] * (max_norm_point - norm_com) / max_norm_point + noise

        return force
    
    def compute_reward(self, force):
        # calculate the reward
        return np.abs(force / self.state['mass'][0]) * np.abs(force / self.state['mass'][0]) + 1 * (np.abs(force - self.state['mass'][0]) < 0.05) - 1 - 0 * (force < 0.1)
    
    def step(self, action):
        # 计算action与candidate_actions的距离
        distances = np.linalg.norm(self.state['candidate_actions'][:, 0:2] - action[0:2], axis=1)
        index = np.argmin(distances)  # 选择最近的候选动作
        action = self.state['candidate_actions'][index]
        force = self.compute_force(action)
        self.state['history'][self.state['attempt']] = np.array([action[0], action[1], action[2], force])
        self.state['attempt'] += 1

        return self.get_observation(), self.compute_reward(force), self.is_done(force), self.is_truncated(force), self.get_info()
    
    def reset(self, seed=None, options=None):
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
            cv2.circle(frame, (int(500 * point[0]), int(500 * point[1])), 3, (0, 255, 0), -1)  # Black point
        # draw the convex
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(500 * self.state['convex'], dtype=np.int32)], (0, 0, 0))
        alpha = 0.1
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # draw the com
        com = self.state['com']
        mass = self.state['mass']
        cv2.circle(frame, (int(500 * com[0]), int(500 * com[1])), 3, (0, 0, 255), -1)  # Red point
        cv2.putText(frame, f'{25 * mass[0]:.1f}', (int(500 * com[0]) + 5, int(500 * com[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # draw the history
        cv2.putText(frame, f"Attempts: {self.state['attempt']}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        for i in range(self.state['attempt']):
            action_x, action_y, _, force = self.state['history'][i]
            if i == self.state['attempt'] - 1:
                cv2.circle(frame, (int(500 * action_x), int(500 * action_y)), 3, (255, 0, 0), -1)  # If the last action, red point
            else:
                cv2.circle(frame, (int(500 * action_x), int(500 * action_y)), 3, (0, 0, 0), -1)  # Black point
            cv2.putText(frame, f'{25 * force:.1f}', (int(500 * action_x) + 5, int(500 * action_y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        if self.render_mode == 'human':
            cv2.imshow('Environment State', frame)
            cv2.waitKey(100)  # wait for 1ms
            return frame
        elif self.render_mode == 'rgb_array':
            return frame
        
    def initialize_state(self, contour, convex):
        # scale the contour to 0-1
        scale = np.max((contour[:, :2].max(axis=0) - contour[:, :2].min(axis=0)))
        scale_point = contour[:, :2].min(axis=0)
        contour = (contour[:, :2] - scale_point) / scale
        convex = (convex - scale_point) / scale
        # calculate the candidate actions
        contour = utils.interpolate_contour(contour)
        contour = utils.add_normal_to_contour(contour)
        candidate_actions = calculate_candidate_actions(contour)
        # initialize the mass and com
        mass = np.array([np.random.uniform(5, 25)]) / 25
        while True:
            com = np.random.uniform(0, 50, 2) / 50
            if mplPath.Path(convex).contains_point(com) and np.linalg.norm(contour[:, 0:2] - com, axis=1).min() > 0.01:
                break
        self.state = {'contour': contour, 'convex': convex, 'candidate_actions': candidate_actions, 'mass': mass, 'com': com, 'attempt': 0, 'history': np.zeros((self.max_steps, 4))}

def calculate_intersections(lines_theta1, lines_theta2):
    # input: lines_theta1: [[primary_intercept1, theta1], [primary_intercept2, theta1]], lines_theta2: [[secondary_intercept1, theta2], [secondary_intercept2, theta2]]
    # output: rectangle: np.array([[pt1], [pt2], [pt3], [pt4]]), pri-pri-sec-sec
    # 计算两条直线的交点
    pts = []
    for line1, line2 in zip([lines_theta1[0], lines_theta1[0], lines_theta1[1], lines_theta1[1]], [lines_theta2[0], lines_theta2[1], lines_theta2[1], lines_theta2[0]]):
        intercept1, theta1 = line1
        intercept2, theta2 = line2
        a1, b1 = np.sin(theta1), -np.cos(theta1)
        a2, b2 = np.sin(theta2), -np.cos(theta2)
        x = (intercept2 * b1 - intercept1 * b2) / (a1 * b2 - a2 * b1)
        y = (intercept1 * a2 - intercept2 * a1) / (a1 * b2 - a2 * b1)
        pts.append([x, y])

    return np.array(pts)

def calculate_rectangles(contour):
    # 将倾角变换到0-180度，注意这里的倾角是法向量倾角，需要先转换为矩形主轴倾角
    thetas = np.mod(contour[:, 2] + np.pi / 2, np.pi)
    # 使用 K-means 聚类算法将倾角聚为两类
    kmeans = KMeans(n_clusters=2, random_state=0).fit(thetas.reshape(-1, 1))
    labels = kmeans.labels_
    indices1, indices2 = np.where(labels == 0)[0], np.where(labels == 1)[0]
    # 找到两类倾角的代表值，使用中位数计算
    theta1, theta2 = np.median(thetas[indices1], axis=0), np.median(thetas[indices2], axis=0)

    # 计算轮廓点在对应倾角上的截距，从而分割为若干离散线段
    intercept_list_theta1, intercept_list_theta2 = [], []
    for indices, theta, intercept_list in zip([indices1, indices2], [theta1, theta2], [intercept_list_theta1, intercept_list_theta2]):
        intercepts = np.dot(contour[indices, 0:2], np.array([-np.sin(theta), np.cos(theta)]))
        Z = linkage(intercepts.reshape(-1, 1), method='single')
        cluster_labels = fcluster(Z, t=0.05, criterion='distance')
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            intercept_indices = np.where(cluster_labels == label)[0]
            intercept_list.append([np.median(intercepts[intercept_indices], axis=0), np.count_nonzero(intercept_indices)])
    
    # 从interception_list中选择个数最多的线段和距离最大个数线段最近的线段作为主线段
    # 从interception_list中选择截距之差最大的两条线段作为副线段
    intercept_list = intercept_list_theta1
    if len(intercept_list) == 2:
        primary_intercept_theta1, secondary_intercept_theta1 = intercept_list, intercept_list
    elif len(intercept_list) == 3:
        max_count_intercept = max(intercept_list, key=lambda x: x[1])
        intercept_list = sorted(intercept_list, key=lambda x: np.abs(x[0] - max_count_intercept[0]), reverse=False)
        primary_intercept_theta1 = intercept_list[:2]
        secondary_intercept_theta1 = [intercept_list[0], intercept_list[2]]
    elif len(intercept_list) == 4:
        intercept_list = sorted(intercept_list, key=lambda x: x[0], reverse=True)
        primary_intercept_theta1, secondary_intercept_theta1 = intercept_list[1:3], [intercept_list[0], intercept_list[3]]
    intercept_list = intercept_list_theta2
    if len(intercept_list) == 2:
        primary_intercept_theta2, secondary_intercept_theta2 = intercept_list, intercept_list
    elif len(intercept_list) == 3:
        max_count_intercept = max(intercept_list, key=lambda x: x[1])
        intercept_list = sorted(intercept_list, key=lambda x: np.abs(x[0] - max_count_intercept[0]), reverse=False)
        primary_intercept_theta2 = intercept_list[:2]
        secondary_intercept_theta2 = [intercept_list[0], intercept_list[2]]
    elif len(intercept_list) == 4:
        intercept_list = sorted(intercept_list, key=lambda x: x[0], reverse=True)
        primary_intercept_theta2, secondary_intercept_theta2 = intercept_list[1:3], [intercept_list[0], intercept_list[3]]
        
    # 分类讨论：2-2，3-3，4-3，3-4，4-4
    if len(intercept_list_theta1) == 2 and len(intercept_list_theta2) == 2:
        # 只有一个矩形2-2，将截距个数较多的倾角作为矩形倾角
        if np.count_nonzero(indices1) > np.count_nonzero(indices2): 
            rectangle1 = calculate_intersections([[primary_intercept_theta1[0][0], theta1], [primary_intercept_theta1[1][0], theta1]], [[secondary_intercept_theta2[0][0], theta2], [secondary_intercept_theta2[1][0], theta2]])
            theta2 = theta1
        else:
            rectangle1 = calculate_intersections([[primary_intercept_theta2[0][0], theta2], [primary_intercept_theta2[1][0], theta2]], [[secondary_intercept_theta1[0][0], theta1], [secondary_intercept_theta1[1][0], theta1]])
            theta1 = theta2
        rectangle2, rectangle_collision = np.array([[0, 0]] * 4), np.array([[0, 0]] * 4)
    else:
        # 有两个矩形3-3，4-3，3-4，4-4，设置碰撞检测区
        rectangle1 = calculate_intersections([[primary_intercept_theta1[0][0], theta1], [primary_intercept_theta1[1][0], theta1]], [[secondary_intercept_theta2[0][0], theta2], [secondary_intercept_theta2[1][0], theta2]])
        rectangle2 = calculate_intersections([[primary_intercept_theta2[0][0], theta2], [primary_intercept_theta2[1][0], theta2]], [[secondary_intercept_theta1[0][0], theta1], [secondary_intercept_theta1[1][0], theta1]])
        collision_offset_theta1 = np.sign(primary_intercept_theta1[0][0] - primary_intercept_theta1[1][0]) * 0.05  # Modified by Yue Wang
        collision_offset_theta2 = np.sign(primary_intercept_theta2[0][0] - primary_intercept_theta2[1][0]) * 0.05
        rectangle_collision = calculate_intersections([[primary_intercept_theta1[0][0] + collision_offset_theta1, theta1], [primary_intercept_theta1[1][0] - collision_offset_theta1, theta1]], [[primary_intercept_theta2[0][0] + collision_offset_theta2, theta2], [primary_intercept_theta2[1][0] - collision_offset_theta2, theta2]])

    return rectangle1, rectangle2, rectangle_collision, theta1, theta2

def calculate_candidate_actions(contour):
    rectangle1, rectangle2, rectangle_collision, theta1, theta2 = calculate_rectangles(contour)
    # 从principle_points1[0]开始，到principle_points1[1]结束, 按照等间隔取线段上的点
    interpolated_points1 = np.linspace((rectangle1[0] + rectangle1[3]) / 2, (rectangle1[1] + rectangle1[2]) / 2, int(np.linalg.norm(rectangle1[0] - rectangle1[1]) / 0.01))
    interpolated_points2 = np.linspace((rectangle2[0] + rectangle2[3]) / 2, (rectangle2[1] + rectangle2[2]) / 2, int(np.linalg.norm(rectangle2[0] - rectangle2[1]) / 0.01))
    candidate_actions = []
    for points in interpolated_points1:
        if not mplPath.Path(rectangle_collision).contains_point(points):
            candidate_actions.append([points[0], points[1], theta1])
    for points in interpolated_points2:
        if not mplPath.Path(rectangle_collision).contains_point(points):
            candidate_actions.append([points[0], points[1], theta2])
    candidate_actions = np.array(candidate_actions)

    return candidate_actions
