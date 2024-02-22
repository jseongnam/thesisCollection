from torch.distributions import Categorical
import gym
from IPython import display # IPython 라이브러리에 있는 display 모듈을 사용합니다.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim,64),
            nn.ReLU(),
            nn.Linear(64,out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # 훈련 모드 설정
    
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def forward(self, x):
        pdparam = self.model(x)
        return pdparam
    
    def act(self, state):
        # print("state:",state)
        # state = np.array(state[:1], dtype=np.float32)
        x = torch.from_numpy(state.astype(np.float32))
        # print("x=",x)
        # print("x.shape:",x.shape)
        pdparam = self.forward(x)
        pd = Categorical(logits = pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()

def train(pi, optimizer):
    # REINFORCE 알고리즘 내부 경사 상승 루프
    T = len(pi.rewards)
    rets = np.empty(T, dtype = np.float32)
    future_ret = 0.0
    # 이득을 효율적으로 계산
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main():
    env = gym.make('CartPole-v0')
    
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim,out_dim)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
        
        # env.step(env.action_space.sample())


    for epi in range(300):
        state = env.reset()
        # print("env.reset() : ",state)
        for t in range(200):
            action = pi.act(state)
            # print('screen:',screen)
            # screen = screen.astype(np.uint8)
            # plt.imshow(screen)

            # display.clear_output(wait=True)
            # display.display(plt.gcf())
            # print(env.step(action))
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            time.sleep(0.02)
            if done :
                break
        
        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()
        print(f'Episode {epi}, loss : {loss},\
            total_reward:{total_reward}, solved:{solved}')
    
if __name__ == '__main__':
    main()