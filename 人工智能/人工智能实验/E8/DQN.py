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


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # TODO write another linear layer here with 
        # inputsize "hidden_size" and outputsize "output_size"
        ...

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        # TODO calculate output with another layer
        ...
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        # TODO Define buffer with given capacity
        ...

    def len(self):
        # TODO Return the size of buffer
        ...

    def push(self, *transition):
        # TODO Add transition to buffer
        ...

    def sample(self, batch_size):
        # TODO Sample transitions from buffer
        ...

    def clean(self):
        # TODO Clean buffer
        ...


class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        # network for evaluate
        self.eval_net = QNet(input_size, hidden_size, output_size)
        # target network
        self.target_net = QNet(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.eps = args.eps
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
    
    def choose_action(self, obs):
        # TODO Return an action according to the given observation "obs"
        ...

    def store_transition(self, *transition):
        self.buffer.push(*transition)
        
    def learn(self):
        # [Epsilon Decay]
        # if self.eps > args.eps_min:
        #     self.eps *= args.eps_decay

        # [Update Target Network Periodically]
        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1
        
        # [Sample Data From Experience Replay Buffer]
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions)  # to use 'gather' latter
        dones = torch.FloatTensor(dones)
        rewards = torch.FloatTensor(rewards)

        # TODO [Calculate and Perform Gradient Descend]
        # For example:
        # 1. calculate q_eval with eval_net and q_target with target_net
        # 2. td_target = r + gamma * (1-dones) * q_target
        # 3. calculate loss between "q_eval" and "td_target" with loss_fn
        # 4. optimize the network with self.optim


def main():
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim)                         # 初始化DQN智能体
    for i_episode in range(args.n_episodes):                            # 开始玩游戏
        obs = env.reset()                                               # 重置环境
        episode_reward = 0                                              # 用于记录整局游戏能获得的reward总和
        done = False
        step_cnt=0
        while not done and step_cnt<500:
            step_cnt+=1
            env.render()                                                # 渲染当前环境(仅用于可视化)
            action = agent.choose_action(obs)                           # 根据当前观测选择动作
            next_obs, reward, done, info = env.step(action)             # 与环境交互
            agent.store_transition(obs, action, reward, next_obs, done) # 存储转移
            episode_reward += reward                                    # 记录当前动作获得的reward
            obs = next_obs
            if agent.buffer.len() >= args.capacity:
                agent.learn()                                           # 学习以及优化网络
        print(f"Episode: {i_episode}, Reward: {episode_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="CartPole-v1",  type=str,   help="environment name")
    parser.add_argument("--lr",             default=1e-3,       type=float, help="learning rate")
    parser.add_argument("--hidden",         default=64,         type=int,   help="dimension of hidden layer")
    parser.add_argument("--n_episodes",     default=500,        type=int,   help="number of episodes")
    parser.add_argument("--gamma",          default=0.99,       type=float, help="discount factor")
    # parser.add_argument("--log_freq",       default=100,        type=int)
    parser.add_argument("--capacity",       default=10000,      type=int,   help="capacity of replay buffer")
    parser.add_argument("--eps",            default=0.0,        type=float, help="epsilon of ε-greedy")
    # parser.add_argument("--eps_min",        default=0.05,       type=float)
    parser.add_argument("--batch_size",     default=128,        type=int)
    # parser.add_argument("--eps_decay",      default=0.999,      type=float)
    parser.add_argument("--update_target",  default=100,        type=int,   help="frequency to update target network")
    args = parser.parse_args()
    main()