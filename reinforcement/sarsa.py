import gym
import numpy as np
import torch
class Sarsa:
    def __init__(self,ODim, ADim):
        self.actValue = torch.zeros(ODim,ADim) # math.sqrt(dim) x math.sqrt(dim) 0으로 된 행렬
        self.reward = torch.zeros(ODim,ADim)
        self.rate = 0.99
    def getReward(self, action,state,reward):
        self.reward[state][action] = reward
    # state
    def train(self,actionB,stateB,actionA,stateA) :
        self.actValue[stateB][actionB] = self.rate * self.actValue[stateA][actionA] + self.reward[stateB][actionB]
    
# env.step : 0,1,2,3 -> 상,하,좌,우
def main():
    env = gym.make("FrozenLake-v0", is_slippery=False)
    # print(env.observation_space.n)
    # print(env.action_space.n)
    sarsa = Sarsa(env.observation_space.n,env.action_space.n)
    for i in range(3000):
        env.reset()
        print(i)
        while 1:
            try:
                actionBefore = actionAfter
                stateBefore = stateAfter

                actionAfter = env.action_space.sample()
                stateAfter, reward, done, _ = env.step(actionAfter)
                sarsa.train(actionBefore,stateBefore, actionAfter, stateAfter)
                env.render()
                # print(f"state:{state},reward:{reward},done:{done},R:{R}")
                if reward == 0 and done == True:
                    sarsa.getReward(actionAfter,stateAfter,-0.01)
                    break
                elif reward == 1 :
                    sarsa.getReward(actionAfter,stateAfter,1)
                    break
            except UnboundLocalError:
                actionAfter = env.action_space.sample()
                stateAfter, reward, done, _ = env.step(actionAfter)
                env.render()
                # print(f"state:{state},reward:{reward},done:{done},R:{R}")
                if reward == 0 and done == True:
                    sarsa.getReward(actionAfter,stateAfter,-0.01)
                    break
                elif reward == 1 :
                    sarsa.getReward(actionAfter,stateAfter,1)
                    break
    print(f"Q : {sarsa.actValue}")

main()
