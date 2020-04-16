import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    
    def __init__(self, n_state, n_action, hidden1, hidden2):
        
        """ Creates the actor network.
        
        (params):
            n_state (int): The size of the state space
            n_action (int): The size of the action space
            hidden1 (int): The size of the first hidden layer
            hidden2 (int): The size of the first hidden layer
            """
        
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_state, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_action)
        self.reset_params()
        
    def forward(self, x):
        
        """ Predicts actions from states """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.out(x))
        return x
    
    def reset_params(self):
        
         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
         self.out.weight.data.uniform_(-3e-3, 3e-3)
    

class Critic(nn.Module):
    
    def __init__(self, n_state, n_action, state_hidden, hidden1, hidden2):
        
        """ Creates the actor network.
        
        (params):
            n_state (int): The size of the state space
            n_action (int): The size of the action space
            state_hidden (int): The number of units in the layer between states
                and concatination
            hidden (list of ints): The size of the hidden layers """
        
        super(Critic, self).__init__()
        self.process = nn.Linear(n_state, state_hidden)
        self.fc1 = nn.Linear(state_hidden+n_action, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)
        self.reset_params()
        
    def forward(self, states, actions):
        
        """ Predicts Q values from states  and actions """
        
        state_process = F.relu(self.process(states))
        x = torch.cat((state_process, actions), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
        return x
    def reset_params(self):
        
         self.process.weight.data.uniform_(*hidden_init(self.process))
         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
         self.out.weight.data.uniform_(-3e-3, 3e-3)
