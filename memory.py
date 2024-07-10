import numpy as np


class ReplayBuffer:
    def __init__(self, input_shape, buffer_size=int(1e6), batch_size=64):
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.states = np.zeros((self.buffer_size, *input_shape))
        self.next_states = np.zeros((self.buffer_size, *input_shape))
        self.actions = np.zeros((self.buffer_size))
        self.rewards = np.zeros((self.buffer_size))
        self.dones = np.zeros((self.buffer_size), dtype=bool)
        self.mem_counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        i = self.mem_counter % self.buffer_size
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.mem_counter += 1

    def sample(self):
        mem_max = min(self.mem_counter, self.buffer_size)
        batch = np.random.choice(mem_max, self.batch_size, replace=False)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        dones = self.dones[batch]

        return states, actions, rewards, next_states, dones
