# Q-learning Update Rule
# Q_α_{t+1}(s, a; ω) is updated as follows:
# If t is in T(s,a)(ω):
# Q_α_{t+1}(s, a; ω) = (1 - α_t) * Q_α_t(s, a; ω) + α_t * (r(s, a) + γ * max_{a′∈A} Q_α_t(S′_t(ω), a′; ω))
# Otherwise:
# Q_α_{t+1}(s, a; ω) = Q_α_t(s, a; ω)

import gym
import numpy as np
import torch

import gym
import numpy as np
import torch
class Q_learning:
    def __init__(self,ODim, ADim):
        self.QValue = torch.zeros(ODim,ADim)
        self.reward = torch.zeros(ODim,ADim)
        self.rate = 0.9
    
        self.model = torch.nn.Sequential(
            torch.nn.Linear(ODim*ADim,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,ODim*ADim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()
    def stepSize(self,time):
        self.alpha = 1/time # stepsize(alpha) function = 1/t -> Robbins-Monro 조건 만족
    def getReward(self, action,state,reward):
        self.reward[state][action] = reward
    def maxQ(self,state):
        answer = -21e8
        for i in range(4):
            if answer < self.QValue[state][i]:
                answer = self.QValue[state][i]
        return answer        
        
    def Qtrain(self,actionB,stateB,stateA) :
        self.QValue[stateB][actionB] = (1-self.alpha) * self.QValue[stateB][actionB] + self.alpha*(self.reward[stateB][actionB] + self.rate*self.maxQ(stateA))
        
    def train(self):
        predicted = self.model(self.QValue.squeeze().view(1,-1)).view(16,4) 
        loss = self.loss_fn(predicted, self.QValue)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.QValue = predicted.detach()
# env.step : 0,1,2,3 -> 상,하,좌,우
def main():
    env = gym.make("FrozenLake-v0", is_slippery=False)

    qLearning = Q_learning(env.observation_space.n,env.action_space.n)
    for i in range(300):
        env.reset()
        print(i)
        stateBefore = 0
        while 1:
            action = env.action_space.sample()
            stateAfter, reward, done, _ = env.step(action)
            env.render()
            if reward == 0 and done == True:
                qLearning.getReward(action,stateBefore,-1)
                break
            elif reward == 1 :
                qLearning.getReward(action,stateBefore,1)
                break
            stateBefore = stateAfter
    time = 1
    while time < 3000:
        env.reset()

        print(i)
        stateBefore = 0
        while 1:
            qLearning.stepSize(time)
            action = env.action_space.sample()
            stateAfter, reward, done, _ = env.step(action)
            qLearning.Qtrain(action,stateBefore, stateAfter)
            env.render()
            if reward == 0 and done == True:
                time += 1 
                break
            
            elif reward == 1 :
                time += 1 
                break
            stateBefore = stateAfter
        qLearning.train()
    print(f"Q : {qLearning.QValue}")
    

main()
