from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Reshape
from tensorflow.keras.activations import sigmoid, tanh, linear, relu
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.losses import mse, hinge

import numpy as np
import matplotlib.pyplot as plt
import random

from EnvWrapper import MsPacmanWrapper, BreakoutWrapper
from ExperienceReplays import PrioritisedExperienceReplay, ExperienceReplay

class DQN():
    
    ''' The basic version of a deep Q-network 
    (params):
        env (Open-ai gym environment or custom wrapper): The environment, 
            should always be a MsPacman Wrapper for now
        memory (Custom Class): The memory; normal or PER'''
    
    def __init__(self, env, memory, learn_rate=0.01, gamma=0.1, epsilon=0.1):
        
        self.env = env
        self.memory = memory
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.learn_rate = learn_rate
        
        self.model = self.create_network()
        self.fill_memory()
        
    def create_network(self):
        
        
        """ Creates the Q-network """
        
        self.width = self.memory.width
        self.height = self.memory.height
        self.channels = self.memory.channels
        self.fps = self.memory.frames_per_state 
        self.actions = self.env.action_space.n

        
        model = Sequential()
        model.add(Conv2D(32, 4, activation = relu, 
                         input_shape = (self.width, self.height,
                                        self.channels*self.fps)))
        model.add(MaxPooling2D((3,3)))
        
        model.add(Conv2D(64, 3, activation = relu))
        model.add(MaxPooling2D((3,3)))
        
        model.add(Conv2D(64, 2, activation=relu))
        model.add(MaxPooling2D((3,3)))
        
        model.add(Flatten())
        model.add(Dense(self.actions*64, activation = relu))
        model.add(Dense(self.actions, activation = linear))
        
        model.compile(optimizer = Adam(learning_rate=self.learn_rate), loss=mse)
        model.summary()
        
        return model
        
    def fill_memory(self):
        
        ''' Fills the memory with actions, to prevent undefined samples'''
        
        self.env.reset()
        
        for i in range(self.memory.capacity):
            action = self.env.action_space.sample()
            state, reward, done, info = self.env.step(action)
            state = state.reshape((self.width, self.height, self.channels))
            self.memory.append(state, action, reward, done, 0.0)
            if done:
                self.env.reset()
            
    def train_on_batch(self, batch_size=2**6):
        
        ''' Trains on a batch of experiences '''
        
        # Get the batch
        states, actions, rewards, new_states, terminals, idxs = self.memory.get_batch(batch_size)
        
        # Normalise inputs
        states = states.astype('float32')
        new_states = new_states.astype('float32')
        states /= 255.0
        new_states /= 255.0
        # Reshape to match input of network
        states = states.reshape((-1,self.width, self.height, self.fps*self.channels))
        new_states = new_states.reshape((-1,self.width, self.height, self.fps*self.channels))
        
        # Predict Q values for states and future states
        target = self.model.predict(states)
        
        # Calculate target Q values
        
        
        for i, Q_vals in enumerate(target):
            Q_old = target[i][int(actions[i])]
            if terminals[i]:
                Q_target = rewards[i]
            else:
                
                Q_future = self.model.predict(new_states[i].reshape(-1, self.width, 
                                                         self.height,
                                                         self.fps*self.channels))
                best_future = np.max(Q_future)
                # Update rule
                Q_target = rewards[i] + self.gamma * (best_future)
            
            target[i][int(actions[i])] = Q_target
        # Train the model
        self.model.train_on_batch(states, target)
        
    def select_action(self, state):
        
        ''' Select the best action from the current state '''
        
        if random.random() < self.epsilon:
            # Then select a random action
            action = self.env.action_space.sample()
            return action
        
        # Get the right shape for the network
        state = state.reshape((1,self.width, self.height, self.fps*self.channels))
        state = state.astype('float32')
        state /= 255.0
        # Predict Q values
        Q_vals = self.model.predict(state)
        #print(Q_vals)
        
        # Get the best and break ties randomly
        action = np.random.choice(np.flatnonzero(Q_vals == Q_vals.max()))
        return action

    def train(self, num_episodes, plot=False, save=False, debug=True):
        
        """ Trains the agent 
        (params):
            num_epsodes (int): The number of episodes
            plot (bool): Whether to plot the performance
            save (bool): Whether to save the agents models
            debug (bool): Whether to print debug information """
        
        history = []
        
        try:
            for episode in range(num_episodes):
                
                # Reset the environment
                state_list = [self.env.reset() for i in range(self.fps)]
                state = np.array(state_list)
                performance = 0
                frames = 0
                while True:
                    frames += 1
                    # Take the action determined by the policy
                    action = self.select_action(state)
                    frame, reward, done, _ = self.env.step(action)
                    
                    # Update the performance
                    performance += reward
                    
                    state_list.pop(0)
                    state_list.append(frame)
                    state = np.array(state_list)
                    
                    
                    # TODO: Sort out TD error stuff
                    self.memory.append(
                        frame.reshape(self.width, self.height, self.channels),
                        action, reward, done, 0)
                    
                    # Update the model
                    if frames % 4 == 0:
                        
                        self.train_on_batch(2**10)
                    
                    if done:
                        print('Episode: ',episode, ', Performance: ', performance, ', Frames: ', frames)
                        history.append(performance)
                        break
        finally:    
            plt.plot(history)
            plt.show()
                
# Code is run from here
if __name__ == '__main__':
    env = BreakoutWrapper(85,85,True)
    memory =  ExperienceReplay(2**16, (85,85,1,2))
    agent = DQN(env, memory, learn_rate=0.0001)
    agent.train(1000)