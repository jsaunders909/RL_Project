import gym
import time
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt

class MsPacmanWrapper():
    
    ''' An environment wrapper for Ms Pacman 
    Used to reduce memory size of frames
    follows the resest/step/render architecture of gym'''
        
    def __init__(self, width, height, use_grayscale=True):
    
        self.env = gym.make('MsPacman-v0')
        self.width = width 
        self.height = height
        self.gray = use_grayscale
        self.action_space = self.env.action_space
    
    def process_frame(self, frame):
        
        if self.gray:
            frame = color.rgb2gray(frame)
        frame = resize(frame, (self.width, self.height))    
        return frame    
    
    def reset(self):
        
        self.env.reset()
        
    def step(self, action):
        
        state, reward, terminal, info = self.env.step(action)
        state = self.process_frame(state)
        return (state, reward, terminal, info)
        
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()
        
env = MsPacmanWrapper(100,100,True)


try:
    env.reset()
    for i in range(1000):
        env.render()
        state, reward, terminal, info = env.step(env.action_space.sample()) # take a random action
        plt.imshow(state)
        plt.show()
        time.sleep(0.1)
finally:
    env.close()
         