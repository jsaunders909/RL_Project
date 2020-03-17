import gym
import time
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt

class MsPacmanWrapper():
    
    ''' An environment wrapper for Ms Pacman 
    Used to reduce memory size of frames
    follows the resest/step/render architecture of gym
    
    (params):
        width (int): The number of pixels for width after resizing
        height (int): The number of pixels for height after resizing
        use_grayscale (bool): Whether to convert to 1-channel grayscale
        '''
        
    def __init__(self, width, height, use_grayscale=True):
    
        self.env = gym.make('MsPacman-v0')
        self.width = width 
        self.height = height
        self.gray = use_grayscale
        self.action_space = self.env.action_space
    
    def process_frame(self, frame):
        
        if self.gray:
            frame = color.rgb2gray(frame)
        frame = resize(255*frame, (self.width, self.height))  
        
        return frame.astype('uint8')    
    
    def reset(self):
        
        return self.process_frame(self.env.reset())
        
    def step(self, action):
        
        state, reward, terminal, info = self.env.step(action)
        state = self.process_frame(state)
        return (state, reward, terminal, info)
        
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()
        
        
class BreakoutWrapper():
    
    ''' An environment wrapper for breakout 
    Used to reduce memory size of frames
    follows the resest/step/render architecture of gym
    
    (params):
        width (int): The number of pixels for width after resizing
        height (int): The number of pixels for height after resizing
        use_grayscale (bool): Whether to convert to 1-channel grayscale
        '''
        
    def __init__(self, width, height, use_grayscale=True):
    
        self.env = gym.make('Breakout-v0')
        self.width = width 
        self.height = height
        self.gray = use_grayscale
        self.action_space = self.env.action_space
    
    def process_frame(self, frame):
        
        frame = frame[8:, 5:-5]
        if self.gray:
            frame = color.rgb2gray(frame)
        frame = resize(255*frame, (self.width, self.height)) 

        return frame.astype('uint8')
    
    def reset(self):
        
        return self.process_frame(self.env.reset())
        
    def step(self, action):
        
        state, reward, terminal, info = self.env.step(action)
        state = self.process_frame(state)
        return (state, reward, terminal, info)
        
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()        
        
env = BreakoutWrapper(85,85,True)

if __name__ == '__main__':
    print(env.action_space)
    try:
        env.reset()
        for i in range(1000):
            env.render()
            state, reward, terminal, info = env.step(env.action_space.sample()) # take a random action
            plt.imshow(state)
            plt.colorbar()
            plt.show()
            time.sleep(0.1)
    finally:
        env.close()
         