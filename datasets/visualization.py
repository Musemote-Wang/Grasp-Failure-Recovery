"""
visualization.py
Created by Yue Wang on 2024-09-10
Version 1.1
数据集可视化
Note:
"""
import matplotlib.pyplot as plt
import numpy as np


def attempts_distribution(data_dir='./data/train_dataset_2d.npz'):
    # 加载数据集
    data = np.load(data_dir)
    attempts = data['attempts']
    dones = data['dones']
    # 绘制尝试次数直方图, 根据dones分为成功和失败两组, 使用统一的bins范围, 每一个bin的正负例使用蓝色和红色紧邻绘制，不要重叠
    plt.figure()
    bins = np.linspace(0.5, 15.5, 16)
    # 统计每个bin的正负例数量
    success, _ = np.histogram(attempts[dones == 1], bins=bins)
    failure, _ = np.histogram(attempts[dones == 0], bins=bins)
    # 绘制直方图
    for i in range(15):
        plt.bar(2.5 * i, success[i], width=1.0, color='b', label='Positive' if i == 0 else None, alpha=0.7)
        plt.bar(2.5 * i + 1, failure[i], width=1.0, color='r', label='Negative' if i == 0 else None, alpha=0.7)
    # 设置x轴刻度为1-15
    plt.xticks(2.5 * np.arange(15) + 0.5, np.arange(1, 16), fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Attempt Distribution', fontsize=14)
    plt.xlabel('Attempt', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linewidth=0.3)
    plt.show()

def force_distribution(data_dir='./data/train_dataset_2d.npz'):
    # 加载数据集
    data = np.load(data_dir)
    forces = data['forces']
    dones = data['dones']
    # 绘制力矩直方图, 根据dones分为成功和失败两组
    plt.figure()
    bins = np.linspace(0, 1, 12)
    # 统计每个bin的正负例数量
    success, _ = np.histogram(forces[dones == 1], bins=bins)
    failure, _ = np.histogram(forces[dones == 0], bins=bins)
    # 绘制直方图
    for i in range(11):
        plt.bar(2.5 * i, success[i], width=1.0, color='b', label='Success' if i == 0 else None, alpha=0.7)
        plt.bar(2.5 * i + 1, failure[i], width=1.0, color='r', label='Failure' if i == 0 else None, alpha=0.7)
    # 设置x轴刻度为0-1, 间隔0.1, 显示一位小数
    plt.xticks(2.5 * np.arange(11) + 0.5, np.round(np.linspace(0, 1, 11), 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Force Distribution', fontsize=14)
    plt.xlabel('Force', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linewidth=0.3)
    plt.show()

if __name__ == '__main__':
    attempts_distribution('./dataset/offline_dataset/train_dataset_v3.npz')
    force_distribution('./dataset/offline_dataset/train_dataset_v3.npz')