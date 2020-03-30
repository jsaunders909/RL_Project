# Deep deterministic policy gradient for Bipedal Walker v3
# keras version = 2.3.0
# tensorflow version = 1.13

# Using parameter noise as suggested here:
# https://openai.com/blog/better-exploration-with-parameter-noise/

# Using OUNoise for actions, code adapted from here:
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

# Referred to the following for guidance:
#   http://arxiv.org/pdf/1509.02971v2.pdf    
#   https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb
#   https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
#   https://github.com/germain-hug/Deep-RL-Keras

import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
import keras.backend as K
import copy
import csv

from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, GaussianNoise, Flatten, Lambda, concatenate
from tensorflow.keras.activations import tanh, relu, linear
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from collections import deque

######################################################################################################### 
# Actor and Critic classes
class PolicyActor:
    
    def __init__(self, state_size, action_size, a_up_limit, batch_size, lr, tau, sess):  
        
        self.sess=sess
        K.set_session(sess)
   
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.a_up_limit = a_up_limit
        self.actor_nn, self.states = self.nn()
        self.target_actor_nn, _ = self.nn()
        
        # The below used to compile and train agent network wrt states and the critic gradients
        self.action_grad=tf.placeholder(tf.float32, [self.batch_size, self.action_size])        
        params_grad=tf.gradients(self.actor_nn.output, self.actor_nn.trainable_weights, -self.action_grad)                
        self.optimize=tf.train.AdamOptimizer(self.lr).apply_gradients(zip(params_grad, self.actor_nn.trainable_weights))        
        
    def nn(self):
        # Actor nn
        
        inp = Input((self.state_size))
        l1 = Dense(256, activation='relu')(inp)
        #l1noise = GaussianNoise(1.0)(l1)
        #l1flat = Flatten()(l1noise)
        #x = Dense(256, activation='relu')(x)
        #x = GaussianNoise(1.0)(x)
        output = Dense(self.action_size, activation='tanh')(l1)

        return Model(inp, output), inp         
    
    def update_target_weights(self):
        # Pass on the weights to the target model at a rate of tau
        
        curr_w = self.actor_nn.get_weights()
        new_w = self.target_actor_nn.get_weights()
        
        for i in range(len(curr_w)):
            new_w[i] = self.tau * curr_w[i] + (1 - self.tau)* new_w[i]
            
        self.target_actor_nn.set_weights(new_w)          
                        
    def train(self, states, grads):
        # Used to compile and train the actor network wrt states and the critic gradients
        self.sess.run(self.optimize, feed_dict={self.states:states,self.action_grad:grads})                  
        
class ValueCritic:
    
    def __init__(self, state_size, action_size, a_up_limit, lr, tau, sess):
        
        self.sess=sess        
        K.set_session(sess)        

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.tau = tau
        self.a_up_limit = a_up_limit
        
        # Initiate critic networks (critic target is initally a copy of critic)
        self.critic_nn, self.critic_states, self.critic_actions = self.nn()
        self.target_critic_nn, _, _ = self.nn()       
        # Holds instance to compute the tf gradients
        self.q_gradients = tf.gradients(self.critic_nn.output, self.critic_actions)[0]        

    def nn(self):
        # Create nn for critics, (2 inputs, one for states other for actions, 3 dense layers 
        # with 256, 256 and 128 neurons with one neuron in output (tells how good the predictions were)
        
        state = Input(batch_shape=(None,self.state_size))
        action = Input(batch_shape=(None,self.action_size))         
        stateL = Dense(256, activation='relu')(state)
        mergedL = concatenate([Flatten()(stateL), action])
        L2 = Dense(256, activation='relu')(mergedL)
        L3 = Dense(128, activation='relu')(L2)        
        output = Dense(1, activation=linear, kernel_initializer=RandomUniform())(L3)
        model = Model([state, action], output)        
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')        

        return model, state, action        
        
    
    def update_target_weights(self):
        # Pass on the weights to the target model at a rate of tau
        
        curr_w = self.critic_nn.get_weights()
        new_w = self.target_critic_nn.get_weights()
        
        for i in range(len(curr_w)):
            new_w[i] = self.tau * curr_w[i] + (1 - self.tau)* new_w[i]
            
        self.target_critic_nn.set_weights(new_w)        


    def compute_grads(self, states, actions):
        # Compute the Q value gradients of states and policy actions for critic using tf
        return self.sess.run(self.q_gradients, feed_dict={self.critic_states:states, self.critic_actions:actions})  

######################################################################################################### 
# Classes for the memory buffer and the noise    
class ExperienceBuffer:
    
    def __init__(self, buffer_cap, batch_size, sess):  
        
        self.buffer = deque(maxlen=buffer_cap)
        self.batch_size = batch_size
        self.sess=sess          
        
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
            
        return (np.array(states), np.array(actions), np.array(rewards), np.array(state_prims), np.array(terminals)) 

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
# DDPG algorithm class    
class DDPG:
    
    def __init__(self, state_size, action_size, a_up_limit, a_bot_limit, actor_lr, critic_lr, tau, gamma, buffer_cap, batch_size, load = False): 
        
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = actor_lr 
        self.critic_lr = critic_lr         
        self.tau = tau
        self.gamma = gamma
        self.a_up_limit = a_up_limit
        self.a_bot_limit = a_bot_limit
        self.buffer_cap = buffer_cap
        self.batch_size = batch_size    

        # Noise to be used with actions        
        self.noise = OUNoise(self.action_size)        
        # Start tf session
        self.sess=tf.Session()        
        # Load the actor and critic models (and their targets)
        self.Actor = PolicyActor(state_size, action_size, a_up_limit, batch_size, actor_lr, tau, self.sess)
        self.Critic = ValueCritic(state_size, action_size, a_up_limit, critic_lr, tau, self.sess)
        
        # If string passed to load param, load the appropriate models (must be in same folder)
        if load != False:           
            
            self.Actor.actor_nn = load_model('a_'+load + '.h5')        
            self.Actor.target_actor_nn = load_model('at_'+load + '.h5')
            self.Critic.critic_nn = load_model('c'+load + '.h5')            
            self.Critic.target_critic_nn = load_model('ct_'+load + '.h5')
            print('* Models loaded successfully *')
               
             
        # Initialise the buffer    
        self.buffer = ExperienceBuffer(self.buffer_cap, self.batch_size, self.sess)

        #self.sess.run(tf.global_variables_initializer())        
        
        
    def step(self, state, action, reward, state_prim, terminal):
        # Take a step, push step to buffer and update models if enough experiences in buffer       
        self.buffer.push(state, action, reward, state_prim, terminal)
        
        if((self.buffer.get_size()) > self.batch_size):
            
            batch = self.buffer.get_batch()
            self.fit(gamma, batch)
            
    def act(self, state):
        # Use actor to predic actions given the state, add noise, and clip to env action bounds
        a = self.Actor.actor_nn.predict(np.expand_dims(state, axis=0))[0]
        a += self.noise.sample() 
        a = np.clip(a, self.a_bot_limit, self.a_up_limit)
        
        return a
    
    def clear_noise(self):
        # Reset noise        
        self.noise.reset()  
        
    def save(self, episode, model_name, startT):
        # Save models for all 4 networks        
        timeTaken = int(time.time() - startT)
        name = model_name + 'ep' + str(episode) + 'Tsec' + str(timeTaken)
        self.Actor.actor_nn.save('a_'+ name + '.h5')          
        self.Critic.critic_nn.save('c_'+ name + '.h5')              
        self.Actor.target_actor_nn.save('at_'+ name + '.h5')  
        self.Critic.target_critic_nn.save('ct_'+ name + '.h5')   
           

    def fit(self, gamma, batch):
        # *** Use batch experiences to update the networks ***
        # Get batch
        states, actions, rewards, state_prims, terminals = batch
        
        # Use the taget actor to predict actions from new states
        target_actor_actions = self.Actor.target_actor_nn.predict(state_prims)   

        # Use critic to criticise the predictions 
        target_Q_values = self.Critic.target_critic_nn.predict([state_prims, target_actor_actions])
        
        # Q targets (bellman)
        Q_targets = []
        
        for tgt in range(len(target_Q_values)):
            Q_targets.append(rewards[tgt] + (gamma * target_Q_values[tgt] * (1 - terminals[tgt])))
            #Q_targets.append(rewards[tgt] + gamma * target_Q_values[tgt])
        
        Q_targets = np.array(Q_targets)
       
        # Train critic
        self.Critic.critic_nn.train_on_batch([states, actions], Q_targets)

        # Actor predicts the actions from the batch states
        actions_pred = self.Actor.actor_nn.predict(states)

        # Use critic to compute Q value gradients under policy from the states and the actions the actor predicted        
        grads = self.Critic.compute_grads(states, actions_pred)
        grads = np.array(grads)

        # Train actor with batch states and the gradients
        self.Actor.train(states, grads)

        # Update target networks with the respective parent networks at a rate of tau
        self.Actor.update_target_weights()
        self.Critic.update_target_weights()
        
#########################################################################################################        
# Main training class        
def training(agent, train_episodes, max_steps, model_name): 

    startT = time.time()       
    episode = 0
    scores, avr, queue_scores = [],[],[]
    
    for episode in range(train_episodes):
    
        state = env.reset()
        agent.clear_noise()        
        score = 0

        for i in range(max_steps):
            
            action = agent.act(state)

            state_prim, reward, terminal, _ = env.step(action)            
            agent.step(state, action, reward, state_prim, float(terminal))
            state = state_prim
            score += reward

            if terminal == True:
                break   
            
        print(score)
        avr.append(score)
        queue_scores.append(score)                
        scores.append([episode,score,np.sum(avr)/len(avr)])            
            
        if (episode % 100 == 0):
            # save model
            agent.save(episode, model_name, startT)
            print('**************************')
            print('Episode: ' + str(episode)) 
            print('100 EP Avr: ' + str((sum(queue_scores)/len(queue_scores)))) 
            print('Alltime average: ' + str(np.sum(avr)/len(avr)))               
            print('**************************')
            
            queue_scores = []
       
    # produce a score graph          
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
    agent.clear_noise()     
    terminal = False
    rewards = 0 
    while terminal == False:    

        env.render()  
        action = agent.act(state)   
        #print(action)        
        state, reward, terminal, info = env.step(action)
        rewards += reward
        if terminal == True:
            print('Reward gained: ',rewards)
            break
        
    env.close()
            
#########################################################################################################
# Env Params
env = gym.make('BipedalWalker-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
a_up_limit = env.action_space.high[0]
a_bot_limit = env.action_space.low[0]

# Agent Params
buffer_cap = int(1e6)
batch_size = 128
gamma = 0.99
tau = 1e-3  
actor_lr = 1e-4 
critic_lr = 3e-4
e = 0
e_decay = 0.998
train_episodes = 2000
max_steps = 700

# Model name (used to save model)
model_name = 'ddpg2'
# False = do not load model, or pass str with name of the saved weights to re-initiate training
# i.e. load_weights = 'ddpg1ep100Tsec527' (without the leading character and .h5 extension)
#load_weights = 'ddpg2ep1000Tsec8345'
load_weights = False

agent = DDPG(state_size, action_size, a_up_limit, a_bot_limit, actor_lr, critic_lr, tau, gamma, buffer_cap, batch_size, load_weights)

# Uncomment one
#training(agent, train_episodes, max_steps, model_name)
testing(agent)

