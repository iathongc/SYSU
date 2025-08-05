# coding=gb2312
# requirements
# - Python >= 3.7
# - torch >= 1.7
# - gym == 0.23
# - (Optional) tensorboard, wandb

import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import matplotlib.pyplot as plt

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)    # 定义第一个全连接层
        self.fc2 = nn.Linear(hidden_size, output_size)   # 定义第二个全连接层

    def forward(self, x):
        x = torch.Tensor(x)      # 将输入数据转换为Tensor
        x = F.relu(self.fc1(x))  # 通过第一个全连接层并应用ReLU激活函数
        x = F.relu(self.fc2(x))  # 通过第二个全连接层并应用ReLU激活函数
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        # 初始化经验回放缓冲区和容量
        self.buffer = []
        self.capacity = capacity

    def len(self):
        # 返回缓冲区当前的大小
        return len(self.buffer)

    def push(self, *transition):
        # 如果缓冲区已满，则移除最早的一个
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        # 将新的转换（transition）添加到缓冲区
        self.buffer.append(transition)

    def sample(self, batch_size):
        # 从缓冲区中随机采样一个batch
        transitions = random.sample(self.buffer, batch_size)
        # 解包采样的转换数据
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def clean(self):
        self.buffer.clear()

class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        # 初始化环境和网络
        self.env = env
        self.eval_net = QNet(input_size, hidden_size, output_size)       # 评估网络
        self.target_net = QNet(input_size, hidden_size, output_size)     # 目标网络
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)  # 优化器
        self.eps = args.eps    # 探索概率
        self.buffer = ReplayBuffer(args.capacity)    # 经验回放缓冲区
        self.loss_fn = nn.MSELoss()                  # 损失函数
        self.learn_step = 0                          # 学习步数

    def choose_action(self, obs):
        # 选择动作
        if np.random.uniform() <= self.eps:
            # 随机选择动作
            action = np.random.randint(0, self.env.action_space.n)
        else:
            # 根据评估网络选择最优动作
            obs = torch.FloatTensor(obs).unsqueeze(0)
            action_values = self.eval_net(obs)
            action = torch.argmax(action_values).item()
        return action

    def store_transition(self, *transition):
        self.buffer.push(*transition)   # 存储转换（transition）

    def learn(self):
        # 训练DQN模型
        if self.eps > args.eps_min:
            # 随着训练逐渐减少探索概率
            self.eps *= args.eps_decay

        if self.learn_step % args.update_target == 0:
            # 每隔固定步数更新目标网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        # 从缓冲区采样
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions)
        dones = torch.FloatTensor(dones)
        rewards = torch.FloatTensor(rewards)

        # 计算Q值
        q_eval = self.eval_net(np.array(obs)).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next = torch.max(self.target_net(np.array(next_obs)), dim=1)[0]
        q_target = rewards + args.gamma * (1 - dones) * q_next

        # 计算损失并反向传播
        dqn_loss = self.loss_fn(q_eval, q_target)
        self.optim.zero_grad()
        dqn_loss.backward()
        self.optim.step()

def main():
    # 创建环境
    env = gym.make(args.env, render_mode='human')
    r = []
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim)
    for i_episode in range(args.n_episodes):
        # 重置环境并初始化参数
        obs = env.reset()[0]
        episode_reward = 0
        done = False
        step_cnt = 0
        while not done and step_cnt < 500:
            step_cnt += 1
            env.render()
            action = agent.choose_action(obs)
            next_obs, reward, done, info, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
            episode_reward += reward
            obs = next_obs
            if agent.buffer.len() >= args.capacity:
                agent.learn()
        print(f"Episode: {i_episode}, Reward: {episode_reward}")
        r.append(episode_reward)

    plt.plot([i for i in range(1, len(r) + 1)], r, color="#4169E1")
    plt.title('Reward')
    plt.show()

    average_reward = []
    for i in range(len(r) - 99):  # 遍历范围为len(r) - 99，确保每次计算100个局数
        sum_reward = sum(r[i:i + 100])  # 计算当前100局的奖励总和
        average_reward.append(sum_reward / 100)  # 将100局的平均奖励添加到average_reward列表

    plt.plot(range(1, len(average_reward) + 1), average_reward, color="#4169E1")
    plt.title('Average reward over 100 episodes')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str)    # 环境名称
    parser.add_argument("--lr", default=1e-3, type=float)            # 学习率
    parser.add_argument("--hidden", default=256, type=int)           # 隐藏层大小
    parser.add_argument("--n_episodes", default=500, type=int)       # 训练轮数
    parser.add_argument("--gamma", default=0.99, type=float)         # 折扣因子
    parser.add_argument("--log_freq", default=100, type=int)         # 日志频率
    parser.add_argument("--capacity", default=5000, type=int)        # 缓冲区容量
    parser.add_argument("--eps", default=1, type=float)              # ε初始值
    parser.add_argument("--eps_min", default=0.05, type=float)       # ε最小值
    parser.add_argument("--batch_size", default=128, type=int)       # 批处理大小
    parser.add_argument("--eps_decay", default=0.999, type=float)    # ε衰减率
    parser.add_argument("--update_target", default=100, type=int)    # 目标网络更新频率
    args = parser.parse_args()
    main()