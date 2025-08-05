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
        self.fc1 = nn.Linear(input_size, hidden_size)    # �����һ��ȫ���Ӳ�
        self.fc2 = nn.Linear(hidden_size, output_size)   # ����ڶ���ȫ���Ӳ�

    def forward(self, x):
        x = torch.Tensor(x)      # ����������ת��ΪTensor
        x = F.relu(self.fc1(x))  # ͨ����һ��ȫ���Ӳ㲢Ӧ��ReLU�����
        x = F.relu(self.fc2(x))  # ͨ���ڶ���ȫ���Ӳ㲢Ӧ��ReLU�����
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        # ��ʼ������طŻ�����������
        self.buffer = []
        self.capacity = capacity

    def len(self):
        # ���ػ�������ǰ�Ĵ�С
        return len(self.buffer)

    def push(self, *transition):
        # ������������������Ƴ������һ��
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        # ���µ�ת����transition����ӵ�������
        self.buffer.append(transition)

    def sample(self, batch_size):
        # �ӻ��������������һ��batch
        transitions = random.sample(self.buffer, batch_size)
        # ���������ת������
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def clean(self):
        self.buffer.clear()

class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        # ��ʼ������������
        self.env = env
        self.eval_net = QNet(input_size, hidden_size, output_size)       # ��������
        self.target_net = QNet(input_size, hidden_size, output_size)     # Ŀ������
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)  # �Ż���
        self.eps = args.eps    # ̽������
        self.buffer = ReplayBuffer(args.capacity)    # ����طŻ�����
        self.loss_fn = nn.MSELoss()                  # ��ʧ����
        self.learn_step = 0                          # ѧϰ����

    def choose_action(self, obs):
        # ѡ����
        if np.random.uniform() <= self.eps:
            # ���ѡ����
            action = np.random.randint(0, self.env.action_space.n)
        else:
            # ������������ѡ�����Ŷ���
            obs = torch.FloatTensor(obs).unsqueeze(0)
            action_values = self.eval_net(obs)
            action = torch.argmax(action_values).item()
        return action

    def store_transition(self, *transition):
        self.buffer.push(*transition)   # �洢ת����transition��

    def learn(self):
        # ѵ��DQNģ��
        if self.eps > args.eps_min:
            # ����ѵ���𽥼���̽������
            self.eps *= args.eps_decay

        if self.learn_step % args.update_target == 0:
            # ÿ���̶���������Ŀ������
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        # �ӻ���������
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions)
        dones = torch.FloatTensor(dones)
        rewards = torch.FloatTensor(rewards)

        # ����Qֵ
        q_eval = self.eval_net(np.array(obs)).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next = torch.max(self.target_net(np.array(next_obs)), dim=1)[0]
        q_target = rewards + args.gamma * (1 - dones) * q_next

        # ������ʧ�����򴫲�
        dqn_loss = self.loss_fn(q_eval, q_target)
        self.optim.zero_grad()
        dqn_loss.backward()
        self.optim.step()

def main():
    # ��������
    env = gym.make(args.env, render_mode='human')
    r = []
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim)
    for i_episode in range(args.n_episodes):
        # ���û�������ʼ������
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
    for i in range(len(r) - 99):  # ������ΧΪlen(r) - 99��ȷ��ÿ�μ���100������
        sum_reward = sum(r[i:i + 100])  # ���㵱ǰ100�ֵĽ����ܺ�
        average_reward.append(sum_reward / 100)  # ��100�ֵ�ƽ��������ӵ�average_reward�б�

    plt.plot(range(1, len(average_reward) + 1), average_reward, color="#4169E1")
    plt.title('Average reward over 100 episodes')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str)    # ��������
    parser.add_argument("--lr", default=1e-3, type=float)            # ѧϰ��
    parser.add_argument("--hidden", default=256, type=int)           # ���ز��С
    parser.add_argument("--n_episodes", default=500, type=int)       # ѵ������
    parser.add_argument("--gamma", default=0.99, type=float)         # �ۿ�����
    parser.add_argument("--log_freq", default=100, type=int)         # ��־Ƶ��
    parser.add_argument("--capacity", default=5000, type=int)        # ����������
    parser.add_argument("--eps", default=1, type=float)              # �ų�ʼֵ
    parser.add_argument("--eps_min", default=0.05, type=float)       # ����Сֵ
    parser.add_argument("--batch_size", default=128, type=int)       # �������С
    parser.add_argument("--eps_decay", default=0.999, type=float)    # ��˥����
    parser.add_argument("--update_target", default=100, type=int)    # Ŀ���������Ƶ��
    args = parser.parse_args()
    main()