import numpy as np
import gc
import time
import cv2

class database:
    def __init__(self, params):
        self.size = params['db_size']
        self.img_scale = params['img_scale']
        self.states = np.zeros([self.size,84,84],dtype='uint8')
        self.actions = np.zeros(self.size, dtype='float32')
        self.terminals = np.zeros(self.size, dtype='float32')
        self.rewards = np.zeros(self.size, dtype='float32')
        self.batch_size = params['batch']
        self.batch_s = np.zeros([self.batch_size,84,84,4])
        self.batch_a = np.zeros([self.batch_size])
        self.batch_t = np.zeros([self.batch_size])
        self.batch_n = np.zeros([self.batch_size,84,84,4])
        self.batch_r = np.zeros([self.batch_size])

        self.counter = 0
        self.flag = False
        return
    
    def get_batches(self):
        for i in range(self.batch_size) :
            idx = 0
            while idx < 3 or (idx > self.counter -2 and idx < self.counter + 3) :
                idx = np.random.randint(3,self.get_size() - 1)
            self.batch_s[i] = np.transpose(self.states[idx-3:idx+1,:,:],(1,2,0)) / self.img_scale
            self.batch_n[i] = np.transpose(self.states[idx-2:idx+2,:,:],(1,2,0)) / self.img_scale
            self.batch_a[i] = self.actions[idx]
            self.batch_t[i] = self.terminals[idx]
            self.batch_r[i] = self.rewards[idx]
        return self.batch_s, self.batch_a, self.batch_t, self.batch_n, self.batch_r
    
    def insert(self, prevstate_proc, reward, action, terminal) :
        self.states[self.counter] = prevstate_proc
        self.rewards[self.counter] = reward
        self.actions[self.counter] = action
        self.terminals[self.counter] = terminal

        self.counter += 1
        if self.counter >= self.size:
            self.flag = True
            self.counter = 0
        return
    def get_size(self):
        if self.flag == False:
            return self.counter
        else:
            return self.size