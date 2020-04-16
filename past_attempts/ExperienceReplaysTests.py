import numpy as np
from ExperienceReplays import *
import random

print()

# First we create a small tree with equal priority
def run_sumtree_tests():
    print('--------------- Equal priority -----------------')
    print()
    print('e = 0.1, eta = 0.1')
    TDErrors = np.zeros((4), dtype='float32')
    tree = SumTree(TDErrors, 0.1,0.1)
    tree.print_tree()
    print()
    
    # If we change e and eta the numbers should change but still have equal priority
    print('e = 0.001, eta = 1' )
    TDErrors = np.zeros((4), dtype='float32')
    tree = SumTree(TDErrors, 0.001,1)
    tree.print_tree()
    print()
    
    # Lets make one of them more important but set eta to 0
    print('Uniform samples as eta = 0')
    TDErrors[2] = 1
    tree = SumTree(TDErrors, 0.001, 0)
    tree.print_tree()
    print()
    
    # Testing e and eta with priority
    print('-------- With Priority on element 2 ---------')
    print('Small e and eta')
    tree = SumTree(TDErrors, 0.01, 0.01)
    tree.print_tree()
    print()
    
    print('Small e and big eta')
    tree = SumTree(TDErrors, 0.01, 2)
    tree.print_tree()
    print()
    
    print('e = 0')
    tree = SumTree(TDErrors, 0, 2)
    tree.print_tree()
    print()
    
    # Next we check if we can update the tree sucessfully
    print('-------------- Updating -----------------')
    print('Before update') 
    tree = SumTree(TDErrors, 0.01, 1)
    tree.print_tree()
    print()
    
    print('Making element 2 have the same priorty again')
    tree.update(2, 0)
    tree.print_tree()
    print()
    
    print('Making element 0 have high priority')
    tree.update(0, 5)
    tree.print_tree()
    print()
    
    
    print('-------------- Sampling ------------------')
    # First we will try to sample from a tree with equal priorities
    print('Equal priorities')
    TDErrors = np.zeros((4), dtype='float32')
    tree = SumTree(TDErrors, 0.1,0.1)
    tree.print_tree()
    samples = tree.sample(1000)
    # Get in a readable dict
    unique, counts = np.unique(samples, return_counts=True)
    counts = dict(zip(unique, counts))
    print(counts)
    print()
    
    print('Element 0 with priority and small eta')
    TDErrors = np.zeros((4), dtype='float32')
    tree = SumTree(TDErrors, 0.1,0.1)
    tree.update(0,1)
    tree.print_tree()
    samples = tree.sample(1000)
    # Get in a readable dict
    unique, counts = np.unique(samples, return_counts=True)
    counts = dict(zip(unique, counts))
    print(counts)
    print()
    
    print('Element 0 with priority and big eta')
    TDErrors = np.zeros((4), dtype='float32')
    tree = SumTree(TDErrors, 0.1,2)
    tree.update(0,1)
    tree.print_tree()
    samples = tree.sample(1000)
    # Get in a readable dict
    unique, counts = np.unique(samples, return_counts=True)
    counts = dict(zip(unique, counts))
    print(counts)
    print()
    



# Here we impliment the naive method of sampling and the sum tree and see the
# time difference with 2^20 ~ 1 million taking 1000 samples

def naive_sample(priorities, num_samples):
    
    ret = []
    total = np.sum(priorities)
    for i in range(num_samples):
        rand = random.random() * total
        cumulative = 0
        for j,priority in enumerate(priorities):
            cumulative += priority
            if cumulative >= rand:
                ret.append(j)
                break
    return ret

def run_time_tests():
    print('---------------- Time Savings ------------------')
    e, eta = 0.1, 0.3
    TDError = np.random.rand(2**12)
    priority = np.power(TDError + e, eta)
    tree = SumTree(TDError, e , eta)
    
    naive_start = time.time()
    naive_sample(priority, 1000)
    naive_time = time.time() - naive_start
    
    tree_start = time.time()
    tree.sample(1000)
    tree_time = time.time() - tree_start

    print('Tree time', tree_time)
    print('Naive Time', naive_time)
    print()

PER = PrioritisedExperienceReplay(capacity = 2**18, state_shape = (2,2,2,4), e = 0.01, eta = 1.5)

for i in range(2**18):
    PER.append(np.array([i+1, i+1, i+1, i+1, i+1, i+1, i+1, i+1]).reshape((2,2,2)), 1, 1, 0, 1 + random.random())
#    PER.debug_print()

#for i in range(50):
#    state, action, reward, new_state, _ = PER.get_batch(1)
#    print(state.reshape(1,-1), new_state.reshape(1, -1))

batch = PER.get_batch(1000)
for i in range(10):
    print(batch[0][i].shape)

