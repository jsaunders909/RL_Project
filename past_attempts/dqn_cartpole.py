# -*- coding: utf-8 -*-
"""DQN_hillcar.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VtSj5pJHt3TnWb_nwMp9Xg7V66l30oNt
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Sequential, load_model
import gym
import random
import time
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline

"""First we create a replay buffer to store previous states."""

class ReplayMemory():

  def __init__(self, capacity):

    self.memory = []
    self.capacity = capacity

  def append(self, state, action, reward, new_state, terminal):

    if len(self.memory) == self.capacity:
      self.memory.pop(0)
      self.memory.append((state, action, reward, new_state, terminal))
    else:
      self.memory.append((state, action, reward, new_state, terminal)) 

  def get_batch(self, batch_size):

    if len(self.memory) < batch_size:
      return
    else:
      return random.sample(self.memory, batch_size)

  def get_size(self):
      return len(self.memory)

class Agent():

  """ An agent that plays the hillcart game using a deep Q network"""
  def __init__(self, env, gamma = 0.99, learn_rate = 0.1, epsilon = 0.1, decay_rate = 1, min_epsilon=0.1, transfer_rate = 1, hidden_layers=[64]):

    self.learn_rate = learn_rate
    self.gamma = gamma

    self.epsilon = epsilon
    self.decay_rate = decay_rate 
    self.transfer_rate = transfer_rate
    self.min_epsilon = min_epsilon

    self.env = env
    self.replay_memory = ReplayMemory(1000)

    self.target = self.create_Q_network(hidden_layers)
    self.model = self.create_Q_network(hidden_layers)

  def create_Q_network(self, hidden_layers):

    """ Creates a deep neural network of dense hidden layers 
    with neurons given by the param hidden_layers"""

    input_size = self.env.observation_space.shape
    model = Sequential()
    model.add(Dense(32, activation='relu',input_shape=input_size))
    for layer in  hidden_layers:
        model.add(Dense(layer, activation='relu'))  
    model.add(Dense(self.env.action_space.n, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learn_rate))
    model.summary()
    return model

  def select_action(self, state, method='epsilon_greedy'):

    """ Selects an action from the action space based on the given method """
    if method=='epsilon_greedy':
      if random.random() < self.epsilon:
        action = env.action_space.sample()
      else:
        Q_pred = self.model.predict(state) 
        action = np.argmax(Q_pred) 
        print('Angle', state[0][2], '-->', Q_pred[0], '-->', action)
      self.epsilon = max(self.min_epsilon, self.epsilon*self.decay_rate)
      return action

  def train_model(self, batch_size):

    """ Uses 'batch_size' samples from recent memory to update Q-value predictions """

    samples = self.replay_memory.get_batch(batch_size)
    if samples == None:
      return
    states = np.array([sample[0] for sample in samples]).reshape(batch_size,-1)

    targets = self.target.predict(states)
    for i,sample in enumerate(samples):
      # Update the target Q values based on predictions
      state,action,reward,new_state,terminal = sample
      if terminal:
        targets[i][action] = reward
      else:
        Q_new_state = self.target.predict(new_state)[0]
        targets[i][action] =  reward + self.gamma * max(Q_new_state)

    # Fit the model to map from states to the target Q values    
    self.model.fit(states, targets, epochs=2, verbose=0)

  def update_target(self):

    """ Brings the target model close to the active model.
    Essentially a soft copy to avoid erratic behaviour """

    weights = self.model.get_weights()
    target_weights = self.target.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = weights[i] * self.transfer_rate + target_weights[i] * (1 - self.transfer_rate)
    self.target.set_weights(target_weights)  
    
  def save_model(self, path):
      self.target.save(path)
      
  def run_loaded(self, path, trials):
      

      agent = load_model(path)

      for trial in range(trials):
          current_state = env.reset().reshape(1,-1)
          
          while(True):
              Q_vals = agent.predict(current_state)
              action = np.argmax(Q_vals)
              new_state, reward, terminal, info = env.step(action)
              env.render(mode='rgb_array')
              current_state = new_state.reshape(1,-1)
              
              if terminal:
                  break
          
      
  def load(self, path):
      
      self.model = load_model(path)
      self.target = load_model(path)
      self.epsilon = self.min_epsilon
    
  def run(self, num_steps):

    """ Runs the agent for num_steps """
    episode_reward = 0
    history = []
    rolling_av = []
    last_5 = []
    episodes_to_save = 10
    episode = 1


    current_state = env.reset().reshape(1,-1)

    for step in range(num_steps):
      action = self.select_action(current_state)

      new_state, reward, terminal, info = env.step(action)
      
      if terminal:
          reward = -1

      episode_reward += reward
      #env.render()

      env.render(mode='rgb_array')


      new_state = new_state.reshape(1,-1)
      self.replay_memory.append(current_state, action, reward, new_state, terminal)
      
      #if step % 5 ==0:
      #   
        
      if terminal:
        self.train_model(min(64, self.replay_memory.get_size())) 
        
        current_state = env.reset().reshape(1,-1)
        
        history.append(episode_reward)
        last_5.append(episode_reward)
        if len(last_5) == 6:
            last_5.pop(0)
            rolling_av.append(sum(last_5)/5)
        
        episodes_to_save -= 1
        episode += 1
        episode_reward = 0 
        self.update_target()
        
        if episodes_to_save == 0:
          plt.plot([i for i in range(1,episode)], history)
          plt.plot([i for i in range(len(rolling_av))], rolling_av) 
          plt.show()
          episodes_to_save = 10
          self.save_model('cartpole_DQN_trial_%s.h5'%(episode))
          
      else:
        current_state = new_state

env = gym.make("CartPole-v1")

observation = env.reset()
print(observation)

RL_agent = Agent(env, hidden_layers=[16], learn_rate= 0.002, epsilon=0.9, decay_rate=0.999, transfer_rate=0.01)
try:
    #RL_agent.load('cartpole_DQN_trial_301.h5')
    RL_agent.run(100000)
finally:
    env.close()
