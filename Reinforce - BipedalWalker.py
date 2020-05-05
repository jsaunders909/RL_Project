# Policy gradient RL - Monte-Carlo variant (REINFORCE)

import gym
import numpy as np
import time
import random
import keras

from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, InputLayer, Reshape, Input, Lambda, Multiply, MaxPool2D
from tensorflow.keras.activations import softmax, tanh, linear, relu, softplus
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.losses import mse, hinge


import matplotlib.pyplot as plt

class PGM():

    def __init__(self, env, train_episodes = 500, e = 0.1, learn_rate=0.01, discount_r = 0.95, load = False):

        self.env = env

        self.inp_space = env.observation_space.shape[0]

        self.a_space = env.action_space.shape[0]

        self.a_up_limit = env.action_space.high[0]

        self.a_bot_limit = env.action_space.low[0]

        self.learn_rate = learn_rate

        self.e = e

        self.discount_r = discount_r

        self.train_episodes = train_episodes

        self.model = self.nn()

        if load != False:
            self.model.load_weights(load)

        self.startT = 0

        print('DISC R: ',discount_r)
        print('LR: ',learn_rate)


    def nn(self):

        opt = Adam(learning_rate=self.learn_rate)

        #create model
        model = Sequential() #add model layers, layer by layer

        #opt = SGD(learning_rate=self.learn_rate)
        model.add(InputLayer(input_shape=(self.inp_space)))
        model.add(Dense(128, activation = 'relu', kernel_initializer='glorot_uniform', bias_initializer=initializers.Constant(value=0.1)))
        model.add(Dense(self.a_space, activation = 'tanh', kernel_initializer='glorot_uniform', bias_initializer=initializers.Constant(value=0.1)))
        model.compile(optimizer = opt, loss='mse')
        model.summary()

        return model

    def discount_and_normalize_rewards(self, episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.discount_r + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        if(np.sum(discounted_episode_rewards) > 0):
            mean = np.mean(discounted_episode_rewards)
            std = np.std(discounted_episode_rewards)
            discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards


    def select_action(self, state):

        # select action, either e-greedy or through nn
        if (random.uniform(0, 1) <= self.e):
            p = ( 2*np.random.random_sample((4))-1)
            #rand = True
        else:
            # get probability of each action from nn
            p = self.model.predict(state, batch_size=1).flatten()
            #rand = False

        return p

    def save(self, episode):
        timeTaken = round((time.time() - self.startT),2)
        name = 'bw1_at_ep ' + str(episode) + '- T (sec) = ' + str(timeTaken)
        self.model.save_weights(name)


    def test_performance(self):

        state = self.env.reset()
        finished = False

        while finished == False:

            # what we see
            self.env.render()

            play_input = state.reshape((1, self.inp_space))
            play_input = play_input.astype('float32')

            act = self.model.predict(play_input, batch_size=1).flatten()

            state, reward, finished, info = env.step(act)
            print(reward)


        env.close()

    def train(self):

        all_r, ep_s, ep_a, ep_r = [],[],[],[]
        total_r, max_r, episode = 0,0,0
        self.startT = time.time()

        for episode in range(self.train_episodes):

            state = self.env.reset()
            total_ep_r = 0
            terminal = False
            #prev_lives = 5

            #self.env.render()

            while terminal == False:

                NN_input = state
                NN_input = NN_input.reshape((1, self.inp_space))
                NN_input = NN_input.astype('float32')

                #select action
                act = self.select_action(NN_input)

                # take action
                state, reward, terminal, info = env.step(act)

                # memorise
                ep_r.append(reward)
                ep_a.append(act)
                ep_s.append(NN_input)

            # sum up all rewards for the episode
            total_ep_r = np.sum(ep_r)

            # record the sum of rewards
            all_r.append(total_ep_r)

            total_r = np.sum(all_r)

            max_r = np.amax(all_r)

            if episode % 1 == 0:

                # Mean reward
                mean_reward = np.divide(total_r, episode+1)
                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", total_ep_r)
                print("Mean Reward", mean_reward)
                print("Max reward so far: ", max_r)


            # update network
            disc_reward=self.discount_and_normalize_rewards(ep_r)

            X = np.squeeze(np.vstack([ep_s]))
            X = X.reshape((X.shape[0], X.shape[1]))

            self.model.fit(x=X, y=np.vstack(ep_a), verbose=0, sample_weight=disc_reward, epochs=1)

            # reset vars
            ep_s, ep_a, ep_r = [],[],[]

            #env.close()
            if episode % 100 == 0:
                if episode != 0:
                    self.save(episode)



if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')

    agent = PGM(env, train_episodes = 20001, e = 0.1, learn_rate=0.001, discount_r = 0.98, load = False)
    agent.train()
    #agent.test_performance()
