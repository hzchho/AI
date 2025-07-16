import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from collections import deque
import random
import matplotlib.pyplot as plt
#计算得分连续达到500的轮数
def get500score(reward):
    max_count=0
    count=0
    for i in range(len(reward)):
        if reward[i]==500:
            count+=1
            max_count=max(max_count,count)
        else:
            count=0
    
    return max_count

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        #deque数据结构存储数据，支持在达到最大容量时自动删除旧元素(最早进入的)。
        #最大容量为capacity
        self.buffer = deque(maxlen=capacity)

    def len(self):
        #返回buffer现有的元素数
        return len(self.buffer)

    def push(self, *transition):
        #添加新的经验到buffer
        self.buffer.append(transition)

    def sample(self, batch_size):
        #从缓冲区随机取batch_size个经验
        transition=random.sample(self.buffer,batch_size)
        #将列表中的元素解压成多个元组并转换成numpy数组
        obs, action, rewards, next_obs, dones=zip(*transition)
        return np.array(obs), np.array(action), np.array(rewards), np.array(next_obs), np.array(dones)

    def clean(self):
        #清空buffer
        self.buffer.clear()

class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        #两个神经网络：评估网络和目标网络
        # network for evaluate
        self.eval_net = QNet(input_size, hidden_size, output_size)
        # target network
        self.target_net = QNet(input_size, hidden_size, output_size)
        #Adam优化器进行优化
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        #初始值eps用于贪婪策略
        self.eps = args.eps
        #缓冲区
        self.buffer = ReplayBuffer(args.capacity)
        #均方损失函数
        self.loss_fn = nn.MSELoss()
        #初始化学习步数计数器
        self.learn_step = 0
    
    def choose_action(self, obs):
        #20%概率随机选择动作（使用贪婪算法后逐渐减小）
        if np.random.uniform() < self.eps:
            action=self.env.action_space.sample()
        else:
            q=self.eval_net(obs)
            action=q.argmax().item()
        
        return action

    #存储一个经验到缓冲区
    def store_transition(self, *transition):
        self.buffer.push(*transition)
        
    def learn(self):
        # [Epsilon Decay]
        if self.eps > args.eps_min:
            self.eps *= args.eps_decay

        # [Update Target Network Periodically]
        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1
        
        # [Sample Data From Experience Replay Buffer]
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions).view(-1,1)  # to use 'gather' latter
        dones = torch.FloatTensor(dones).view(-1,1)
        rewards = torch.FloatTensor(rewards).view(-1,1)
        #计算当前obs下的评估网络预测的Q值
        # 【eval神经网络结果为两个动作，与actions聚合后得到动作】
        q_eval=self.eval_net(obs).gather(1,actions)#q_eval为256x1
        #下一个obs对应的目标网络的最大Q值，并用detach()从计算图中分离，以避免反向传播对目标网络的影响。
        #优化
        max_action=self.eval_net(next_obs).max(1)[1].view(-1,1)
        q_target=self.target_net(next_obs).gather(1,max_action)
        #初始未优化
        #q_target=self.target_net(next_obs).max(1)[0].view(-1,1).detach()#q_target为256x1
        td_target=rewards+args.gamma*(1-dones)*q_target
        
        loss=self.loss_fn(q_eval,td_target)
        
        self.optim.zero_grad()#PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        loss.backward()#反向传播
        self.optim.step()

def main():
    env = gym.make(args.env)
    reward_set=[]
    aver_reward_set=[]
    o_dim = env.observation_space.shape[0]
    #print(o_dim)
    a_dim = env.action_space.n
    #print(a_dim)
    agent = DQN(env, o_dim, args.hidden, a_dim)                         # 初始化DQN智能体
    for i_episode in range(args.n_episodes):                            # 开始玩游戏
        obs = env.reset()                                            # 重置环境
        episode_reward = 0                                              # 用于记录整局游戏能获得的reward总和
        done = False
        step_cnt=0
        while not done and step_cnt<500:
            step_cnt+=1
            #env.render()                                                # 渲染当前环境(仅用于可视化)
            action = agent.choose_action(obs)                           # 根据当前观测选择动作
            next_obs, reward, done, info = env.step(action)             # 与环境交互
            agent.store_transition(obs, action, reward, next_obs, done) # 存储转移
            # 当buffer满时清空
            # if agent.buffer.len()== args.capacity:
            #     agent.buffer.clean()
                
            episode_reward += reward                                    # 记录当前动作获得的reward
            obs = next_obs
            if agent.buffer.len() >= 256:
                agent.learn()                                           # 学习以及优化网络
        print(f"Episode: {i_episode}, Reward: {episode_reward}")
        reward_set.append(episode_reward)
        aver_reward_set.append(sum(reward_set[len(reward_set)-100:len(reward_set):])/100)

    plt.plot([i for i in range(len(reward_set))],reward_set)
    plt.title("Reward")
    plt.show()
    print("最多连续：",get500score(reward_set),"局得分达到500分")
    plt.plot([i for i in range(len(aver_reward_set))],aver_reward_set)
    plt.axhline(y=475, color='r', linestyle='--', label='y = 475')
    plt.legend()
    #plt.plot([i for i in range(len(aver_reward_set))],[475 for i in range(len(aver_reward_set))],c='g')
    plt.title("Average_reward")
    plt.ylim(0,500)
    plt.show()
    print("百局平均中是否有超过475分:",(max(aver_reward_set)>=475))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="CartPole-v1",  type=str,   help="environment name")
    parser.add_argument("--lr",             default=1e-3,       type=float, help="learning rate")
    parser.add_argument("--hidden",         default=128,         type=int,   help="dimension of hidden layer")
    parser.add_argument("--n_episodes",     default=500,        type=int,   help="number of episodes")
    parser.add_argument("--gamma",          default=0.99,       type=float, help="discount factor")
    # parser.add_argument("--log_freq",       default=100,        type=int)
    parser.add_argument("--capacity",       default=1024,      type=int,   help="capacity of replay buffer")
    parser.add_argument("--eps",            default=0.08,        type=float, help="epsilon of ε-greedy")
    parser.add_argument("--eps_min",        default=0.05,       type=float)
    parser.add_argument("--batch_size",     default=128,        type=int)
    parser.add_argument("--eps_decay",      default=0.999,      type=float)
    parser.add_argument("--update_target",  default=100,        type=int,   help="frequency to update target network")
    args = parser.parse_args()
    main()