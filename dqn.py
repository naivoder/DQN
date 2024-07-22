import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from memory import ReplayBuffer

def lr_lambda(epoch):
    if epoch < 20000:
        return 3e-4
    else:
        return 1e-4


class ActionValue(torch.nn.Module):
    def __init__(self, input_shape, n_actions, alpha=3e-4, chkpt_file="weights/dqn.pt"):
        super(ActionValue, self).__init__()
        self.chkpt_file = chkpt_file

        self.conv1 = torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1_input_dim = self._calculate_fc1_input_dim(input_shape)
        self.fc1 = torch.nn.Linear(self.fc1_input_dim, 512)
        self.out = torch.nn.Linear(512, n_actions)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=alpha)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.loss = torch.nn.MSELoss()  # use squared l1 instead of mse?

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.out(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

    def _calculate_fc1_input_dim(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        x = torch.nn.functional.relu(self.conv1(dummy_input))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        return x.numel()


class DQNAgent:
    def __init__(
        self,
        env_name,
        input_shape,
        n_actions,
        alpha=3e-4,
        gamma=0.99,
        eps_min=0.01,
        eps_dec=5e-7,
        batch_size=64,
        mem_size=100000,
        replace_target_count=1000,
    ):
        self.gamma = gamma
        self.epsilon = 1.0
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.replace_target_count = replace_target_count
        self.counter = 0

        self.memory = ReplayBuffer(input_shape, int(mem_size), batch_size)
        self.q = ActionValue(
            input_shape, n_actions, alpha, f"weights/{env_name}_dqn.pt"
        )
        self.target_q = ActionValue(
            input_shape, n_actions, alpha, f"weights/{env_name}_target_dqn.pt"
        )

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.q.device)
            actions = self.q(state)
            return torch.argmax(actions).item()

        return np.random.randint(0, self.n_actions)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        if self.counter % self.replace_target_count == 0:
            self.update_target_parameters()

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.FloatTensor(states).to(self.q.device)
        actions = torch.IntTensor(actions).to(self.q.device)
        next_states = torch.FloatTensor(next_states).to(self.q.device)
        rewards = torch.FloatTensor(rewards).to(self.q.device)
        dones = torch.BoolTensor(dones).to(self.q.device)

        self.q.optimizer.zero_grad()
        # get Q value for chosen actions, need np.arange for proper indexing
        q_pred = self.q(states)[np.arange(self.batch_size), actions]

        # max returns tuple of max_val, index
        target_vals = self.target_q(next_states).max(dim=1)[0]
        target_vals[dones] = 0.0

        q_target = rewards + self.gamma * target_vals

        loss = self.q.loss(q_target, q_pred).to(self.q.device)
        loss.backward()
        self.q.optimizer.step()

        self.counter += 1
        # self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

    def update_target_parameters(self):
        self.target_q.load_state_dict(dict(self.q.named_parameters()))

    def save_checkpoint(self):
        self.q.save_checkpoint()
        self.target_q.save_checkpoint()

    def load_checkpoint(self):
        self.q.load_checkpoint()
        self.target_q.load_checkpoint()
