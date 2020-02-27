from collections import deque
from random import choices


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.queue = deque(maxlen=capacity)

    def record(self, obs, action, reward, new_obs, done):
        entry = (obs, action, reward, new_obs, done)
        self.queue.append(entry)

    def get_batch(self, batch_size):
        sample = choices(self.queue, k=batch_size)
        out_dict = {'o':[],'a':[],'r':[],'o2':[],'d':[]}
        for o, a, r, o2, d in sample:
            out_dict['o'].append(o)
            out_dict['a'].append(a)
            out_dict['r'].append(r)
            out_dict['o2'].append(o2)
            out_dict['d'].append(d)
        return out_dict

    def size(self):
        return len(self.queue)
