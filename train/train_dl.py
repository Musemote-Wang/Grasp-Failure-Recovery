"""
train.py
Created by Yue Wang on 2024-10-10
Version 1.0
模型训练
Note:
"""
import os
import argparse
import json
import matplotlib.pyplot as plt
import models
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='GraspEnv_v3', help='environment name')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--model', type=str, default='Dirnet', help='model type')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
parser.add_argument('--observation_dim', type=int, default=16, help='hidden dimension')
parser.add_argument('--num_labels', type=int, default=10000, help='20000, 10000, 5000, 2000, 1000, 500')
args = parser.parse_args()
log_dir = f'./checkpoint/{args.env}/{args.model}/lr_{args.lr}_bs_{args.batch_size}_hid_{args.hidden_dim}'
os.makedirs(log_dir, exist_ok=True)
# 保存配置文件
config_path = os.path.join(log_dir, 'config.json')
with open(config_path, 'w') as f:
    json.dump(vars(args), f, indent=4)

def calculate_loss(model, criterion, batch):
    observations, actions, next_observations, dones, forces, attempts = batch
    if args.model == 'Segnet':
        predictions = model(observations, actions, attempts)
        loss, loss_dir = criterion(predictions, dones), criterion(predictions, dones)
    if args.model == 'Transnet':
        predictions = model(observations, actions)
        loss, loss_dir = criterion(predictions, dones), criterion(predictions, dones)
    if args.model == 'Dirnet':
        predictions = model(observations, actions)
        loss, loss_dir = criterion(predictions, dones), criterion(predictions, dones)
    if args.model == 'Curnet':
        next_states, pred_next_states, pred_measures, pred_actions, direct_measures = model(observations, actions, next_observations)
        loss_measure = criterion(forces, pred_measures)
        loss_forward = criterion(next_states, pred_next_states)
        loss_inverse = criterion(actions, pred_actions)
        loss = 0*loss_measure + 10 * loss_forward + 1 * loss_inverse
        loss_dir = criterion(forces, direct_measures)
    if args.model == 'Vnet-cur':
        next_states, pred_next_states, pred_measures, pred_actions, direct_measures = model[0](observations, actions, next_observations)
        predictions = model[1](torch.cat([pred_next_states, direct_measures], dim=1))  # S1+S2
        # predictions = model(direct_measures)  # S1
        # predictions = model(pred_next_states)  # S2
        # predictions = model(torch.cat([pred_next_states, infos[:, 0:1]], dim=1))  # accurate S1
        loss, loss_dir = criterion(predictions, dones), criterion(predictions, dones)

    return loss, loss_dir

def train_epoch(model, criterion, optimizer, train_loader):
    model.train()
    train_loss, train_loss_dir = [], []
    for batch in train_loader:
        optimizer.zero_grad()
        loss, loss_dir = calculate_loss(model, criterion, batch)
        if torch.isnan(loss):
            print(f"NaN loss detected at train epoch")
            continue
        loss.backward()
        if args.model == 'curnet':
            loss_dir.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 梯度裁剪
        optimizer.step()
        train_loss.append(loss.item())
        train_loss_dir.append(loss_dir.item())
    
    return np.mean(np.array(train_loss)), np.mean(np.array(train_loss_dir))

def eval_epoch(model, criterion, eval_loader):
    model.eval()
    eval_loss, eval_loss_dir = [], []
    for batch in eval_loader:
        loss, loss_dir = calculate_loss(model, criterion, batch)
        eval_loss.append(loss.item())
        eval_loss_dir.append(loss_dir.item())

    return [np.mean(np.array(eval_loss)), np.std(np.array(eval_loss))], [np.mean(np.array(eval_loss_dir)), np.std(np.array(eval_loss_dir))]

def train(model, criterion, optimizer, scheduler, train_loader, eval_loader, epochs):
    train_losses, eval_losses_mean, eval_losses_std = [], [], []
    for epoch in range(epochs):
        train_loss, train_loss_dir = train_epoch(model, criterion, optimizer, train_loader)
        eval_loss, eval_loss_dir = eval_epoch(model, criterion, eval_loader)
        train_losses.append([train_loss, train_loss_dir])
        eval_losses_mean.append([eval_loss[0], eval_loss_dir[0]])
        eval_losses_std.append([eval_loss[1], eval_loss_dir[1]])
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss[0]:.4f}+-{eval_loss_dir[1]:.4f}")
        scheduler.step()
    
    return train_losses, eval_losses_mean, eval_losses_std

def main():
    # 加载离线数据集
    train_data = np.load('./dataset/offline_dataset/train_dataset_v3.npz')
    eval_data = np.load('./dataset/offline_dataset/eval_dataset_v3.npz')
    # 创建数据集: (observations, actions, next_observations, dones, forces, attempts)
    train_dataset = TensorDataset(
        torch.tensor(train_data['observations'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(train_data['actions'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(train_data['next_observations'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(train_data['dones'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(train_data['forces'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(train_data['attempts'], dtype=torch.float32).to(device='cuda'))
    train_dataset = Subset(train_dataset, range(args.num_labels))
    eval_dataset = TensorDataset(
        torch.tensor(eval_data['observations'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(eval_data['actions'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(eval_data['next_observations'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(eval_data['dones'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(eval_data['forces'], dtype=torch.float32).to(device='cuda'), 
        torch.tensor(eval_data['attempts'], dtype=torch.float32).to(device='cuda'))
    # 划分训练集和验证集
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=int(len(eval_dataset) / 10), shuffle=False)
    # 初始化模型
    if args.model == 'Segnet':
        model = models.SegmentNet().to(device='cuda')
    if args.model == 'Transnet':
        model = models.TransformerNet(observation_dim=args.observation_dim, hidden_dim=args.hidden_dim).to(device='cuda')
    if args.model == 'Dirnet':
        model = models.DirectNet(observation_dim=args.observation_dim, hidden_dim=args.hidden_dim).to(device='cuda')
    if args.model == 'Curnet':
        model = models.CuriosityNet(observation_dim=16, hidden_dim=args.hidden_dim).to(device='cuda')
    if args.model == 'Vnet-cur':
        model = [
            models.CuriosityNet(observation_dim=16, hidden_dim=args.hidden_dim).to(device='cuda'), 
            models.ValueNet(input_dim=args.hidden_dim+1).to(device='cuda')]
        # 加载训练的curnet模型
        model[0].load_state_dict(torch.load(f'./checkpoint_label/curnet/labels_20000_H1+H2/model.pth'))
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    if args.model == 'Vnet-cur':
        optimizer = optim.Adam(model[1].parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    # 训练模型
    train_losses, eval_losses_mean, eval_losses_std = train(model, criterion, optimizer, scheduler, train_loader, eval_loader, epochs=200)
    # 保存模型和训练结果
    losses = {'train_losses': np.array(train_losses), 'eval_losses_mean': np.array(eval_losses_mean), 'eval_losses_std': np.array(eval_losses_std)}
    np.save(os.path.join(log_dir, 'loss.npy'), losses)
    torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
    print("模型已保存到 model.pth")

if __name__ == '__main__':
    main()
