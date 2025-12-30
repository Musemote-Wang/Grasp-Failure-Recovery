"""
GraspAgent_dl
Created by Yue Wang on 2024-11-10
Version 1.1
config = {'env': 'GraspEnv_v1/v2/v3', 'model': 'Random/'SegmentNet/TransformerNet/CuriosityNet/DirectNet'}
Note:
env在这里只起到一个数据容器的作用, 保存候选抓取。在执行一次抓取后, 通过实际反馈更新env, 用于返回观测。不用env内部的step函数更新
"""

import envs
import gymnasium
import matplotlib.pyplot as plt
import models
import numpy as np
import torch

class GraspAgent_dl:
    def __init__(self, config):
        self.env = gymnasium.make(id=config['env'], render_mode='rgb_array')
        self.model = load_model(config)

    def calculate_scores(self):
        candidate_actions = torch.tensor(self.env.state['candidate_actions'], dtype=torch.float32).to(device='cuda')
        observation = torch.tensor(self.env.get_observation(), dtype=torch.float32).to(device='cuda')
        observations = torch.cat([observation.unsqueeze(0)] * candidate_actions.shape[0], dim=0)
        with torch.no_grad():
            output = self.model(observations, candidate_actions)
        scores = output.squeeze().cpu().numpy()
        self.scores = np.clip(scores, 0, 1)
        
        return np.clip(scores, 0, 1)
    
    def choose_action(self, num_topk=100):
        scores = self.calculate_scores()
        topk = np.argsort(scores)[-num_topk:]
        topk_probs = scores[topk] / np.sum(scores[topk])
        action = self.env.state['candidate_actions'][np.random.choice(topk, p=topk_probs)]

        return [action]

    def update(self, action, force):
        self.env.state['history'][self.env.state['attempt']] = np.array([action[0], action[1], action[2], force])
        self.env.state['attempt'] += 1

    def reset(self, contour, convex):
        self.env.reset()
        self.env.initialize_state(contour, convex)
        

def load_model(config):
    env_id, model_id = config['env'], config['model']
    if model_id == 'Random':
        return None
    elif model_id == 'Segnet':
        model = models.SegmentNet()
    elif model_id == 'Transnet':
        model = models.TransformerNet()
    elif model_id == 'Curnet':
        model = models.CuriosityNet()
    elif model_id == 'Dirnet':
        model = models.DirectNet()
    model.load_state_dict(torch.load(f'./checkpoint/{env_id}/{model_id}/model_best.pth'))
    model.to(device='cuda')
    model.eval()
    
    return model