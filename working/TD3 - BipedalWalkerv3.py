# Twin Delayed Deep Deterministic Policy Gradient (TD3) for Bipedal Walker v3

#   Refer to the following for guidance:
#   https://arxiv.org/pdf/1802.09477.pdf 
#   https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
#   https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
#   https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal


import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import copy
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

#########################################################################################################
# Actor and Critic classes

# Fan initialisation https://github.com/udacity/deep-reinforcement-learning
def fan_in_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class PolicyActor(nn.Module):
    
    def __init__(self, state_size, action_size):
        
        super(PolicyActor, self).__init__()

        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)        
        self.init_params()    

    def init_params(self):
        self.fc1.weight.data.uniform_(*fan_in_init(self.fc1))
        self.fc1.weight.data.uniform_(*fan_in_init(self.fc2))        
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))        
        return F.tanh(self.fc3(l2))

class ValueCritic(nn.Module):

    def __init__(self, state_size, action_size):

        super(ValueCritic, self).__init__()
        
        self.fc_state1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        self.init_params()

    def init_params(self):
        self.fc_state1.weight.data.uniform_(*fan_in_init(self.fc_state1))
        self.fc2.weight.data.uniform_(*fan_in_init(self.fc2))
        #self.fc3.weight.data.uniform_(*fan_in_init(self.fc3))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        l1_s_a = torch.cat([state, action], 1)        
        #l1state = F.leaky_relu(self.fc_state1(state))
        #l2 = torch.cat((l1state, action), dim=1)
        l2 = F.relu(self.fc_state1(l1_s_a))
        l3 = F.relu(self.fc2(l2))
        return self.fc3(l3)


#########################################################################################################
# Classes for the memory buffer and the noise
class ExperienceBuffer:

    def __init__(self, buffer_cap, batch_size):

        self.buffer = deque(maxlen=buffer_cap)
        self.batch_size = batch_size

    def push(self, state, action, reward, state_prim, terminal):
        # Add experience to buffer
        exp = tuple((state, action, reward, state_prim, terminal))
        self.buffer.append(exp)

    def get_batch(self):
        # Return a batch of experiences
        states, actions, rewards, state_prims, terminals = [],[],[],[],[]

        batch = random.sample(self.buffer, self.batch_size)

        for exp in batch:
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            state_prims.append(list(exp[3]))
            terminals.append([exp[4]])
        
        states = torch.FloatTensor(np.vstack(np.array(states))).to(device)
        actions = torch.FloatTensor(np.vstack(np.array(actions))).to(device)
        rewards = torch.FloatTensor(np.vstack(np.array(rewards))).to(device)
        state_prims = torch.FloatTensor(np.vstack(np.array(state_prims))).to(device)
        terminals = torch.FloatTensor(np.vstack(np.array(terminals))).to(device)          

        return (states,actions,rewards,state_prims,terminals)

    def get_size(self):
        # Return size of buffer
        return len(self.buffer)

# Adapted from: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise():

    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size

        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

#########################################################################################################
# TD3 algorithm class
class TD3():

    def __init__(self, state_size, action_size, actor_lr, critic_lr, tau, gamma, l2_w_decay, a_up_limit, a_bot_limit, buffer_cap, batch_size, model_name, load):

        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.gamma = gamma
        self.a_up_limit = a_up_limit
        self.a_bot_limit = a_bot_limit
        self.model_name = model_name
        self.action_size = action_size
        self.batch_size = batch_size

        # Noise to be used with actions
        self.noise = OUNoise(self.action_size)

        # Load the actor and critic models (and their targets)
        self.Actor = PolicyActor(state_size, action_size).to(device)
        self.Actor_Target = PolicyActor(state_size, action_size).to(device)
        self.Actor_Optimizer = optim.Adam(self.Actor.parameters(), lr=actor_lr)       
        
        self.Critic_1 = ValueCritic(state_size, action_size).to(device)
        self.Critic_1_Target = ValueCritic(state_size, action_size).to(device)
        #self.Critic_1_Optimizer = optim.Adam(self.Critic_1.parameters(), lr=critic_lr, weight_decay=l2_w_decay) 
        self.Critic_1_Optimizer = optim.Adam(self.Critic_1.parameters(), lr=critic_lr) 
        
        self.Critic_2 = ValueCritic(state_size, action_size).to(device)
        self.Critic_2_Target = ValueCritic(state_size, action_size).to(device)
        #self.Critic_2_Optimizer = optim.Adam(self.Critic_2.parameters(), lr=critic_lr, weight_decay=l2_w_decay)         
        self.Critic_2_Optimizer = optim.Adam(self.Critic_2.parameters(), lr=critic_lr)         

        # If string passed to load param, load the appropriate models (must be in folder named after the model)
        if load != False:
            
            self.Actor.load_state_dict(torch.load(self.model_name+'/'+ 'a_'+load + '.pth'))
            self.Critic_1.load_state_dict(torch.load(self.model_name+'/'+ 'c1_'+load + '.pth'))
            self.Critic_2.load_state_dict(torch.load(self.model_name+'/'+ 'c2_'+load + '.pth'))
            self.Actor_Target.load_state_dict(torch.load(self.model_name+'/'+ 'at_'+load + '.pth'))
            self.Critic_1_Target.load_state_dict(torch.load(self.model_name+'/'+ 'c1t_'+load + '.pth'))
            self.Critic_2_Target.load_state_dict(torch.load(self.model_name+'/'+ 'c2t_'+load + '.pth'))
                       
            print('* Models loaded successfully *')

        # Initialise the buffer
        self.buffer = ExperienceBuffer(buffer_cap, batch_size)

    def step(self, state, action, reward, state_prim, terminal, update):
        # Take a step, push step to buffer and update models if enough experiences in buffer
        self.buffer.push(state, action, reward, state_prim, terminal)

        if((self.buffer.get_size()) > self.batch_size):

            batch = self.buffer.get_batch()
            self.fit(batch, update)

    def act(self, state, noise):
        # Use actor to predic actions given the state, add noise, and clip to env action bounds
        state = torch.from_numpy(state).float().to(device)
        action = self.Actor(state).cpu().data.numpy()
        if (noise == True):
            #action += self.noise.sample() # Use OU noise
            action = action + np.random.normal(0, 0.1, size=self.action_size) # Use Normal noise 
        return np.clip(action, self.a_bot_limit, self.a_up_limit)

    def clear_noise(self):
        # Reset noise
        self.noise.reset()

    def save(self, episode, model_name, startT):
        # Save models for all 6 networks
        timeTaken = int(time.time() - startT)
        name = model_name + 'ep' + str(episode) + 'Tsec' + str(timeTaken)
        
        torch.save(self.Actor.state_dict(), model_name+'/'+ 'a_'+ name + '.pth')
        torch.save(self.Critic_1.state_dict(), model_name+'/'+ 'c1_'+ name + '.pth')
        torch.save(self.Critic_2.state_dict(), model_name+'/'+ 'c2_'+ name + '.pth')       
        torch.save(self.Actor.state_dict(), model_name+'/'+ 'at_'+ name + '.pth')
        torch.save(self.Critic_1.state_dict(), model_name+'/'+ 'c1t_'+ name + '.pth')
        torch.save(self.Critic_2.state_dict(), model_name+'/'+ 'c2t_'+ name + '.pth')
        
    def transfer_weights(self, parent_nn, target_nn):
        # Transfer weights to target networks
        for new_w, curr_w in zip(target_nn.parameters(), parent_nn.parameters()):
            new_w.data.copy_(self.tau*curr_w.data + (1.0-self.tau)*new_w.data)

    def fit(self, batch, update):
        # *** Use batch experiences to update the networks ***
        # Get batch
        states, actions, rewards, state_prims, terminals = batch    

        # Use the taget actor to predict clipped actions + normal nose from new states
        norm_noise = torch.ones_like(actions).data.normal_(0, 0.2).to(device)
        norm_noise = norm_noise.clamp(-0.5, 0.5)    
        target_actor_actions = (self.Actor_Target(state_prims) + norm_noise)
        target_actor_actions = target_actor_actions.clamp(self.a_bot_limit, self.a_up_limit)        

        # Use target critics to criticise actor and compute target Q
        Critic_1_Target_Q = self.Critic_1_Target(state_prims, target_actor_actions)
        Critic_2_Target_Q = self.Critic_2_Target(state_prims, target_actor_actions)
        # Use the targets of the critic which has the smallest values
        Target_Q = torch.min(Critic_1_Target_Q, Critic_2_Target_Q) 
        # Compute Q targets (bellman)
        Q_targets = rewards + (self.gamma * Target_Q * (1 - terminals)).detach()
        
        # Train critic 1
        C1_CurrQ = self.Critic_1(states, actions)
        # loss
        C1_loss = F.mse_loss(C1_CurrQ, Q_targets)
        # minimise
        self.Critic_1_Optimizer.zero_grad()
        C1_loss.backward()
        self.Critic_1_Optimizer.step()        

        # Train critic 2
        C2_CurrQ = self.Critic_2(states, actions)
        # loss
        C2_loss = F.mse_loss(C2_CurrQ, Q_targets)
        # minimise
        self.Critic_2_Optimizer.zero_grad()
        C2_loss.backward()
        self.Critic_2_Optimizer.step() 
        
        # Delay policy update
        if(update == True):
            # actor loss
            A_actions = self.Actor(states)
            A_loss = -self.Critic_1(states, A_actions).mean()            
            # minimise
            self.Actor_Optimizer.zero_grad()
            A_loss.backward()
            self.Actor_Optimizer.step()  
            # Transfer weights            
            self.transfer_weights(self.Actor, self.Actor_Target)
            self.transfer_weights(self.Critic_1, self.Critic_1_Target)
            self.transfer_weights(self.Critic_2, self.Critic_2_Target)            
            
#########################################################################################################
# Main training class
def training(agent, train_episodes, max_steps, model_name, update_delay):
    
    startT = time.time()
    episode = 0
    bestScore = 0
    scores, avr, queue_scores = [],[],[]

    for episode in range(train_episodes):

        state = env.reset()
        score = 0

        for i in range(max_steps):

            action = agent.act(state, noise=True)

            state_prim, reward, terminal, _ = env.step(action)
            
            # Update policy net every x time steps
            if (i % update_delay == 0):
                update = True
            else:
                update = False
            
            agent.step(state, action, reward, state_prim, float(terminal), update)
            state = state_prim
            score += reward

            if terminal == True:
                break

        print('Ep: ' + str(episode) + '| reward = ' + str(score))
        avr.append(score)
        queue_scores.append(score)
        scores.append([episode,score,np.sum(avr)/len(avr)])

        if (episode % 50 == 0):
            # save model and print summary every x episodes
            agent.save(episode, model_name, startT)
            print('**************************')
            print('Episode: ' + str(episode))
            print('100 EP Avr: ' + str((sum(queue_scores)/len(queue_scores))))
            print('Alltime average: ' + str(np.sum(avr)/len(avr)))
            print('**************************')

            queue_scores = []

    # produce the score graph
    scores = np.array(scores)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(avr)+1), avr, label = 'Scores')
    ax.plot([i[2] for i in scores], label="Average Score")
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.legend(loc="upper left")
    plt.show()

    # write scores to csv
    file = open("log.csv", 'w')
    with file:
       writer = csv.writer(file)
       writer.writerows(scores)

#########################################################################################################
# Test learned agent
def testing(agent):
    state = env.reset()
    #agent.clear_noise()
    terminal = False
    rewards = 0
    while terminal == False:

        env.render()
        action = agent.act(state, noise=False)
        #print(action)
        state, reward, terminal, info = env.step(action)
        rewards += reward
        if terminal == True:
            print('Reward gained: ',rewards)
            break

    env.close()

#########################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Env Params
env = gym.make('BipedalWalker-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
a_up_limit = env.action_space.high[0]
a_bot_limit = env.action_space.low[0]

# Reproducibility
#seed = 10
#env.seed(seed)
#torch.manual_seed(seed)
#np.random.seed(seed)
#os.environ['PYTHONHASHSEED']=str(seed)
#random.seed(seed)

# Agent Params
buffer_cap = int(1e6)
batch_size = 100
gamma = 0.99
l2_w_decay = 0.0001
tau = 0.005
actor_lr = 0.001
critic_lr = 0.001
train_episodes = 2501
max_steps = 700
policy_net_update_delay = 2

# Model name (used to save model)
model_name = 'td32'
# False = do not load model, or pass str with name of the saved weights to re-initiate training
# i.e. load_weights = td32ep1350Tsec3548 (without the leading character and .pth extension)
load_weights = False

agent = TD3(state_size, action_size, actor_lr, critic_lr, tau, gamma, l2_w_decay, a_up_limit, a_bot_limit, buffer_cap, batch_size, model_name, load_weights)

# Uncomment one
#training(agent, train_episodes, max_steps, model_name, policy_net_update_delay)
testing(agent)
