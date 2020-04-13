import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import gym

GAMMA = 0.99
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
MEMORY = int(1e6)
BATCH_SIZE = 128
TAU = 1e-3
EXPLORATION_NOISE = 0.1
NOISE_CLIP = 0.5

from AC_torch import Actor, Critic



class ReplayBuffer():
    
    def __init__(self, capacity, batch_size, n_state, n_action):
        
        self.batch_size = batch_size
        self.memory = deque(maxlen = capacity)
        self.n_state = n_state
        self.n_action = n_action
        
    def add(self, state, action, reward, new_state, done):
        
        experience = (state, action, reward, new_state, done)
        self.memory.append(experience)
        
    def get_batch(self):
        
        idx = [np.random.randint(0, len(self.memory)) for i in range(self.batch_size)]
        
        states = np.array([self.memory[i][0] for i in idx], dtype = 'float32').reshape(-1, self.n_state)
        actions = np.array([self.memory[i][1] for i in idx], dtype = 'float32').reshape(-1,self.n_action)
        rewards = np.array([self.memory[i][2] for i in idx], dtype = 'float32').reshape(-1,1)
        new_states = np.array([self.memory[i][3] for i in idx], dtype = 'float32').reshape(-1,self.n_state)
        terminals = np.array([self.memory[i][4] for i in idx], dtype = 'float32').reshape(-1,1)
        
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        new_states = torch.from_numpy(new_states).float()
        terminals = torch.from_numpy(terminals).float()
        
        return (states, actions, rewards, new_states, terminals)
        

class DDPG():
    
    def __init__(self, env, n_action, n_state, max_steps = 1000):
        
        self.env = env
        
        self.actor = Actor(n_state, n_action, 256, 512)
        self.actor_target = Actor(n_state, n_action, 256,512)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        
        self.critic = Critic(n_state, n_action, 256,512,128)
        self.critic_target = Critic(n_state, n_action, 256, 512, 128)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        self.memory = ReplayBuffer(MEMORY, BATCH_SIZE, n_state, n_action)
        self.max_steps = max_steps
        
        
    def step(self, state, action, reward, new_state, done):
        
        # Add the experience to memory
        self.memory.add(state, action, reward, new_state, done)
        
        if len(self.memory.memory) > BATCH_SIZE:
            
            batch = self.memory.get_batch()
            self.learn_from_batch(batch)

    def learn_from_batch(self, batch):
        
        # Unpack the experience
        states, actions, rewards, new_states, dones = batch
        
        # ------------------------ Critic Update ---------------------------# 
        
        # Compute best actions from new states
        mu_prime = self.actor_target(new_states)
        
        # Estimate the value of these state action pairs
        Q_prime = self.critic_target(new_states, mu_prime)
        
        # Targets for Q updates
        Q_target = rewards + (1-dones)*GAMMA*Q_prime
        
        # Update the critic
        Q_pred = self.critic(states, actions)
        
        c_loss = F.mse_loss(Q_pred, Q_target)
        self.critic_optim.zero_grad()
        c_loss.backward()
        self.critic_optim.step()
        
         # ------------------------ Actor Update ---------------------------#
        
        mu = self.actor(states)
        a_loss = -self.critic(states, mu).mean()
        
        self.actor_optim.zero_grad()
        a_loss.backward()
        self.actor_optim.step()
        
        self.update_targets()
        
    def update_targets(self):

         for target_param, model_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU*model_param.data + (1.0-TAU)*target_param.data)        
         for target_param, model_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU*model_param.data + (1.0-TAU)*target_param.data)  
            
    def select_action(self, state, add_noise=True):
        
        state = torch.from_numpy(state).float()
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state)
        self.actor.train()
        actions = actions.numpy()
        if add_noise:
            noise = np.random.normal(0, EXPLORATION_NOISE, actions.shape)
            noise.clip(-NOISE_CLIP, NOISE_CLIP)
            actions += noise
            actions.clip(-2,2)
            
        return actions
    
    def train(self, num_episodes):
        
        history = []
        smoothed = []
        
        for i in range(num_episodes):
            
            performance = 0
            state = self.env.reset()
            
            for j in range(self.max_steps):
                
                action = self.select_action(state)
                new_state, reward, done, _ = self.env.step(action)
                performance += reward
                self.step(state, action, reward, new_state, done)
                
                if done:
                    break 
                else:
                    state = new_state
            
            print(performance)
            history.append(performance)
            if i % 20 == 0:
                if i == 0:
                    smoothed.append(performance)
                else:
                    smoothed.append(sum(history[-20:])/20)
                plt.plot(history)
                plt.plot([20*j for j in range(len(smoothed))],smoothed)
                plt.show()
                torch.save(self.actor.state_dict(), 'model.pt')
                
    def test(self, num_episodes):
        
        performance = []
        
        self.actor.load_state_dict(torch.load('model.pt'))
        self.actor.eval()
        
        for i in range(num_episodes):
            
            tot = 0
            state = self.env.reset()
            for j in range(self.max_steps):
                
                action = self.select_action(state, False)
                state, reward, done, _ = self.env.step(action)
                self.env.render()
                tot += reward
                if done:
                    break
                
            print(tot)
            performance.append(tot)
            
        print('Average : %s'%(sum(performance)/len(performance)))

env = gym.make('BipedalWalker-v3')
            
agent = DDPG(env, 4, 24)
#agent.train(2000)
agent.test(5)            