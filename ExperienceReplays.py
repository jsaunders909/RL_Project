# Imports

import numpy as np
import random
import time
from tensorflow import keras
import matplotlib.pyplot as plt

# First we define a batch, we use (state action reward state done TD) structure
# As frames are consecutive, and a state is four frames, we internally store...
#... state and new state together as 5 frames

class ReplayBatch():
    
    def __init__(self, batch_size):
        
        """ A batch used for training
        
            (params): 
                batch_size (int): The size of the batch """
            
        self.batch_size = batch_size


# Next we define the prioritised experience replay class
# We need a sum tree to do this effieciently

class SumTreeNode():
    
    def __init__(self, left_child, right_child, parent,
                 value, leaf = False, root = False, data = None):
        
        """ A node of a sum tree
        (params):
            left_child (SumTreeNode) : The left child 
            right_child (SumTreeNode): The right child 
            parent (SumTreeNode): The parent
            value (float32): The sum of the two children 
            leaf (bool): Is this a leaf
            root (bool): Is this the root
            data: The data stored in  the leaves """
            
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent
        self.value = value
        self.leaf = leaf
        self.data = data
        
    def sample_one(self, passed_value):
        
        """ Selects a random sample from the tree according to priority weight
            using recursion 
            
            (params):
                passed_value (float32): The value to sample from, carried 
                    down to the next level """
                    
        if self.leaf:
            return self.data
        elif passed_value <= self.left_child.value:
            return self.left_child.sample_one(passed_value)
        else:
            passed_value -= self.left_child.value
            return self.right_child.sample_one(passed_value)
    
    def propagate_change(self, change):
        
        """ Passes a change in priority up the tree to ensure correctness 
        
        (paramas):
            change (float32): The change in priority value to be propagted"""
            
        self.value += change
        if self.parent is None:
            return
        else:
            self.parent.propagate_change(change)
        

class SumTree():
    
    def __init__(self, TDErrors, e, eta):
    
        """ Creates a sum tree for sampling in our prioritised experience replay
        by storing values of (TD error + e)^eta and creating a binary search 
        tree where nodes keep track of the cumulative sum of their childrens
        values, we can sample in O(log(n)) steps.
        
        (params): 
            TDErrors (np vector): The vector of TD errors for each state
                action pair in the replay buffer.
            e (float32): A small positive value to give all experiences a 
                non-zero probability of selection
            eta (float32): The prioritisation coefficent, must be >= 0, 
                if eta = 0 then this is uniform sampling, else the greater the
                value the more priorty is given
                
        """
        
        self.priorities = np.power(TDErrors + e, eta)
        
        self.e = e
        self.eta = eta
        
        self.nodes = []
        for i,priority in enumerate(self.priorities):
            # Creates the leaves, we will add parents later
            self.nodes.append(SumTreeNode(None, None, None, priority, 
                                          True, data = i))
            
        # Now layer by layer we create new nodes from the next layer down    
        self.nodes = [self.nodes]
        while len(self.nodes[-1]) != 2: # Up until the second to last layer
            layer = []
            for i in range(0,len(self.nodes[-1]),2):
                left, right = self.nodes[-1][i], self.nodes[-1][i+1]
                top = (SumTreeNode(left,right, None, 
                                   left.value + right.value))
                layer.append(top)
                left.parent = top
                right.parent = top
            self.nodes.append(layer)
        
        left,right = self.nodes[-1][0], self.nodes[-1][1]
        self.root = (SumTreeNode(left,right, None, 
                                 left.value + right.value, root = True))
        left.parent = self.root
        right.parent = self.root
        self.nodes.append([self.root])
    

    def update(self, index, TDError):
        
        """ Updates the tree for a change in TD error
        
            (params):
                index (int): The index at which change has occured
                TDError (float32): The new error """
                
        leaf = self.nodes[0][index]
        old_priority = leaf.value
        new_priority = ((TDError + self.e) ** self.eta)
        change = new_priority -  old_priority   
        leaf.propagate_change(change)
    

    def sample(self, num_samples):
        
        """ Samples from the tree weighted by priority 
        
        (params):
            num_sample (int): The number of samples to collect """
            
        samples = []
        for i in range(num_samples):
            rand = random.random() * self.root.value
            samples.append(self.root.sample_one(rand))
        return samples
        
    def print_recursive(self, node, indent):
        
        if node is None:
            return
        
        print('    '* indent + str(node.value))
        self.print_recursive(node.left_child, indent + 1)
        self.print_recursive(node.right_child, indent + 1)
        
    def print_tree(self):
         
        self.print_recursive(self.root, 0)
         
                
        
            
class PrioritisedExperienceReplay():

    def __init__(self,capacity, state_shape, e = 0.01, eta = 0.5):

        """ A memory buffer that stores previously experienced SARS sequences
        
            (params):
                capacity (int): The number of experiences to store this should 
                    be a power of two for best efficiency 
                frame_shape (tuple of ints): The shape of an individual state
                (WIDTH, HEIGHT, CHANNELS, FRAMES/STATE) """
                    
        self.capacity = capacity            

        self.width = state_shape[0]
        self.height = state_shape[1]
        self.channels = state_shape[2]
        self.frames_per_state = state_shape[3]
        
        
        # We store all frames as a continuous array, there is also an extra 
        # buffer at the beginning so that if frame 0 is selected it is possible
        # to select the previous frames
        
        # e.g. with frames 0,1,2,3,4,5, and 3 frames per state the memeory array would be
        # [3,4,5,0,1,2,3,4,5] so selecting the state, new state pair [3,4,5,0]
        # from the transition 5 -> 0 is possible
        
        self.buffer_length = self.frames_per_state
        
        self.frames = np.zeros((capacity + self.buffer_length, self.width, self.height, 
                                self.channels), dtype='uint8') # Everything
        # should be in raneg 0-255 and normalised later

        self.tree = SumTree(np.zeros(capacity),e,eta)
        
        self.actions = np.zeros(capacity)
        self.rewards = np.zeros(capacity)
        self.terminal = np.zeros(capacity)
        
         # This should be between 0 and capacity
        self.current_index = 0
        
    def append(self, new_state, action, reward, terminal, TDError):
        
        
        ''' Adds a step to the replay memory '''
        
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.terminal[self.current_index] = terminal
        
        self.tree.update(self.current_index, TDError)
        self.frames[self.current_index + self.buffer_length] = new_state
        
        if self.current_index >= self.capacity - self.buffer_length:
            # Add to the buffer
            self.frames[self.current_index - (self.capacity - self.buffer_length)] = new_state
            
        
        self.current_index = (self.current_index + 1) % (self.capacity)
    
    def get_state_new_state_frames(self, i):
        
        ''' Gets the n frames defining the state with index i, 
        as well as the extra frame defining state i + 1
        
        E.g. with 3 frames per state we would get
        
        |1234| where (123) is one state and (234) the immediate next'''
        #print('Index', i)
        #print('Frames: %s to %s'%(self.buffer_length + i - (self.frames_per_state), self.buffer_length + i + 1))
        return self.frames[self.buffer_length + i - (self.frames_per_state): self.buffer_length + i + 1]
    
    
    def update(self, ids, TDs):
        
        ''' Updates the TD values for some experiences '''
        for i,idx in enumerate(ids):
            self.tree.update(idx,TDs[i])
    
    def get_batch(self, batch_size):
        
        ''' Get a batch of experiences
        
        (params): batch_size (int): The size of the batch
        
        (returns): (tuple of np arrays) where:
            index 0 is a collection of states
            index 1 is a collection of actions taken from the corresponding state
            index 2 is the rewards for the actions
            index 3 is the new states arrived in by taking action a from state s
            index 4 is if the new state is terminal '''
            
        idx = self.tree.sample(batch_size)
        
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        terminals = self.terminal[idx]
        states, new_states = [],[]
        
        for i in range(batch_size):
            frames = self.get_state_new_state_frames(idx[i])
            states.append(frames[:-1])
            new_states.append(frames[1:])
            
        states = np.array(states)
        new_states = np.array(new_states)
        return (states, actions, rewards, new_states, terminals, idx)
        
    def debug_print(self):
        
        frames = '|'
        
        for i,frame in enumerate(self.frames):
            
            frames += str(int(frame[0][0][0]))
            if i == self.buffer_length - 1:
                frames += '|'
                
        print(frames)        
        
        
        
class ExperienceReplay():

    """ A memory buffer that stores previously experienced SARS sequences

    (params):
        capacity (int): The number of experiences to store this should 
            be a power of two for best efficiency 
        frame_shape (tuple of ints): The shape of an individual state
        (WIDTH, HEIGHT, CHANNELS, FRAMES/STATE) """
    
    def __init__(self,capacity, state_shape):
                    
        self.capacity = capacity            

        self.width = state_shape[0]
        self.height = state_shape[1]
        self.channels = state_shape[2]
        self.frames_per_state = state_shape[3]
        
        
        # We store all frames as a continuous array, there is also an extra 
        # buffer at the beginning so that if frame 0 is selected it is possible
        # to select the previous frames
        
        # e.g. with frames 0,1,2,3,4,5, and 3 frames per state the memeory array would be
        # [3,4,5,0,1,2,3,4,5] so selecting the state, new state pair [3,4,5,0]
        # from the transition 5 -> 0 is possible
        
        self.buffer_length = self.frames_per_state
        
        self.frames = np.zeros((capacity + self.buffer_length, self.width, self.height, 
                                self.channels), dtype='uint8') 

        
        self.actions = np.zeros(capacity)
        self.rewards = np.zeros(capacity)
        self.terminal = np.zeros(capacity)
        
         # This should be between 0 and capacity
        self.current_index = 0
        
    def append(self, new_state, action, reward, terminal, TDError):
        
        
        ''' Adds a step to the replay memory '''
        
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.terminal[self.current_index] = terminal
        self.frames[self.current_index + self.buffer_length] = new_state
        
        if self.current_index >= self.capacity - self.buffer_length:
            # Add to the buffer
            self.frames[self.current_index - (self.capacity - self.buffer_length)] = new_state
            
        
        self.current_index = (self.current_index + 1) % (self.capacity)
    
    def get_state_new_state_frames(self, i):
        
        ''' Gets the n frames defining the state with index i, 
        as well as the extra frame defining state i + 1
        
        E.g. with 3 frames per state we would get
        
        |1234| where (123) is one state and (234) the immediate next'''
        #print('Index', i)
        #print('Frames: %s to %s'%(self.buffer_length + i - (self.frames_per_state), self.buffer_length + i + 1))
        return self.frames[self.buffer_length + i - (self.frames_per_state): self.buffer_length + i + 1]
    
    def update_TD(self, ids, TDs):
        pass
    
    def get_batch(self, batch_size):
        
        ''' Get a batch of experiences
        
        (params): batch_size (int): The size of the batch
        
        (returns): (tuple of np arrays) where:
            index 0 is a collection of states
            index 1 is a collection of actions taken from the corresponding state
            index 2 is the rewards for the actions
            index 3 is the new states arrived in by taking action a from state s
            index 4 is if the new state is terminal
            index 5 are the idicies '''
            
        idx = []
        for i in range(batch_size):
            idx.append(random.randint(0,self.capacity-1))
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        terminals = self.terminal[idx]
        
        states, new_states = [],[]
        
        for i in range(batch_size):
            frames = self.get_state_new_state_frames(idx[i])
            states.append(frames[:-1])
            new_states.append(frames[1:])
            
        states = np.array(states)
        new_states = np.array(new_states)
        return (states, actions, rewards, new_states, terminals, idx)
        
    def debug_print(self):
        
        frames = '|'
        
        for i,frame in enumerate(self.frames):
            
            frames += str(int(frame[0][0][0]))
            if i == self.buffer_length - 1:
                frames += '|'
                
        print(frames)                