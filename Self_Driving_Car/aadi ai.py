# AI Self Driving Car with pytourch < 1.4
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    def forword(self, state):
        x = f.relu(self.fc1(state))
        q_values = (self.fc2(x))
        return q_values


class ReplayMemory(object):
   
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    def sample(self,batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class dqn():
    
    
    def __init__(self,input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameter(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    def select_action(self, state):
        probs = F.Softmax(self.model(Variable(state, volatile = True))*7) #t=7
        action = probs.muntinomial()
        return action.data[0,0]
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_11_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = 0)
        self.optimizer.step()
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(self.last_state, new_state, torch.LongTensor([int (self.last_action)]),torch.Tensor (self.last_action))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window))+1
    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('loading the model')
            checkpoint = torch.load('last_brain.pth')
            self.modle.load_state_dict(checkpoint('state_dict'))
            self.modle.load_state_dict(checkpoint('optimizer'))
            print('done')
        else:
            print('no checkpoint found')