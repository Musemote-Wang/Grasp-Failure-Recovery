"""
Note:
在调用step和render前, 需要先调用reset初始化环境
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import matplotlib.pyplot as plt
import numpy as np
import envs
import envs.utils as utils
import gymnasium
import agents


def test_env_v1():
    env = gymnasium.make(id='GraspEnv_v1', render_mode='rgb_array')
    attempts, returns, success = np.zeros((100)), np.zeros((100)), np.zeros((100))
    video_writer = cv2.VideoWriter(filename='video.mp4', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=5, frameSize=(500, 500))
    for i in range(10):
        env.reset()
        while True:
            # 随机选择一个动作
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            frame = env.render()  # 获取渲染帧
            # 将帧写入视频文件
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if done or truncated:
                attempts[i] = env.state['attempt']
                success[i] = done and not truncated
                break
    
    # 释放视频写入对象并关闭环境
    video_writer.release()
    cv2.destroyAllWindows()

    print(f'Average Attempts: {attempts.mean()}')
    print(f'Success Rate: {np.sum(success) / 100}')

def test_env_v2_and_v3():
    # agent = graspagents.GraspAgent_dl({'env': 'GraspEnv_v3', 'model': 'Transnet'})
    agent = agents.GraspAgent_bayes({'env': 'GraspEnv_v3'})
    # agent = graspagents.GraspAgent_rl({'env': 'GraspEnv_v3', 'model': 'SAC'})
    num_tasks, num_trials = 1000, 1
    attempts, returns, success = np.zeros((num_tasks * num_trials)), np.zeros((num_tasks * num_trials)), np.zeros((num_tasks * num_trials))
    w = []
    video_writer = cv2.VideoWriter(filename='video.mp4', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=5, frameSize=(500, 500))
    for i in range(num_tasks):
        while True:
            contour, convex = utils.generate_contour()
            agent.reset(contour, convex)
            if agent.env.state['candidate_actions'].shape[0] > 0:
                break
        for j in range(num_trials):
            agent.env.state['attempt'], agent.env.state['history'] = 0, np.zeros((agent.env.max_steps, 4))
            while True:
                # 随机选择一个动作
                # action = agent.env.state['candidate_actions'][np.random.randint(0, len(agent.env.state['candidate_actions']))]
                action = agent.choose_action()[0]
                # print(f'Action: {action}')
                # print(f'mu: {agent.mu}')
                # print(agent.sigma)
                next_observation, reward, done, truncated, info = agent.env.step(action)
                returns[num_trials * i + j] += reward
                agent.env.state['attempt'] -= 1
                agent.update(action, agent.env.state['history'][agent.env.state['attempt'], -1])
                frame = agent.env.render()  # 获取渲染帧
                # save the frame
                cv2.imwrite(f'./frames/{i}_{j}_{agent.env.state["attempt"]}.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # env.compute_force(action)
                # 将帧写入视频文件
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if done or truncated:
                    attempts[num_trials * i + j] = agent.env.state['attempt']
                    success[num_trials * i + j] = done and not truncated
                    print(f'Task {i + 1}, Trial {j + 1}, Attempts: {agent.env.state["attempt"]}, Success: {done and not truncated}')
                    break
            # if done and not truncated:
            #     w.append(agent.scores)    
    
    # 释放视频写入对象并关闭环境
    video_writer.release()
    cv2.destroyAllWindows()
    # w = np.array(w)
    print(f'Average Attempts: {attempts.mean()}')
    print(f'Average Returns: {returns.mean()}')
    print(f'Success Rate: {100 * np.sum(success) / (num_tasks * num_trials)}%')
    plt.plot(attempts)
    plt.savefig('attempts.png')


if __name__ == '__main__':
    test_env_v2_and_v3()
