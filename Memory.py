from collections import namedtuple, deque
import random


# Named tuple transition to group together each value needed in a transition for sampling.
transition = namedtuple('transition',('state', 'action', 'next_state', 'reward', 'done'))

# Memory class for easy storing and retrieving of transitions
class Memory():
    def __init__(self, capacity: int):
        # Deque with a storage capacity so that old transitions are eventually "forgotten" by the agent.
        self.memory = deque(list(), maxlen=capacity)
    
    def save(self, *args):
        # Save 1 transition into the memoty
        self.memory.append(transition(*args))
    
    def sample(self, batch: int):
        # Sample #batch number of transitions from the memory. Randomly to make them decorrelate.
        return random.sample(self.memory, batch)