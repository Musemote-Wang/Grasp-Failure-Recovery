import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np


# 绘制训练曲线
def sb3_draw_eval_curve(log_dir):
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    results = load_results(log_dir)
    x, y = ts2xy(results, 'timesteps')
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    # 绘制 mean reward
    axs[0].plot(np.convolve(np.array(y[:-400:2]), np.ones(250)/250, mode='valid'), color='b')
    axs[0].set_title('Mean Reward', fontsize=14)
    axs[0].set_ylim(-6.1, 0.1)
    axs[0].set_xlabel('Episodes', fontsize=12)
    # axs[0].set_ylabel('Reward')
    # 绘制 mean episode length
    mean_ep_length = results['l']
    axs[1].plot(np.convolve(np.array(mean_ep_length[:-400:2]), np.ones(250)/250, mode='valid'), color='b')
    axs[1].set_title('Mean Episode Length', fontsize=14)
    axs[1].set_ylim(2.1, 8.5)
    axs[1].set_xlabel('Episodes', fontsize=12)
    # axs[1].set_ylabel('Episode Length')
    # 绘制 success rate
    success_rate = []
    for i in range(len(mean_ep_length) // 200 - 3):
        sample_list = mean_ep_length[200 * i : 200 * (i + 1)]
        success_rate.append(len([x for x in sample_list if x < 9]) / 200)
    axs[2].plot(success_rate, color='b')
    axs[2].set_title('Success Rate', fontsize=14)
    axs[2].set_ylim(0.03, 1.03)
    axs[2].set_xlabel('100*Episodes', fontsize=12)
    # axs[2].set_ylabel('Success Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'train_curve.svg'))
    plt.close()


if __name__ == '__main__':
    log_dir = './checkpoint/GraspEnv_v3/rl/SAC/lr_0.0001_bs_512'
    sb3_draw_eval_curve(log_dir)
