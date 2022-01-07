import random

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical

class ExperienceBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_idx = 0
        self.buffer_size = buffer_size
    
    def add(self, e):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(e)
            self.buffer_idx += 1
        else:
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
            self.buffer[self.buffer_idx] = e
    
    def sample(self, batch_size):
        exps = random.sample(self.buffer, k = batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in exps if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in exps if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in exps if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in exps if e is not None])).float()
        terminate = torch.from_numpy(np.vstack([e[4] for e in exps if e is not None])).float()
        return (states, actions, rewards, next_states, terminate)
    
    def get_size(self):
        return len(self.buffer)

class QModel(nn.Module):
    def __init__(self, n_input_dims, n_actions, hidden_units_list = [32]):
        super().__init__()
        n_prev_hu = n_input_dims
        self.hidden_layers = nn.ModuleList([])
        for n_hu in hidden_units_list:
            self.hidden_layers.append(nn.Linear(n_prev_hu, n_hu))
            n_prev_hu = n_hu
        self.output_layer = nn.Linear(n_prev_hu, n_actions)
    
    def forward(self, states):
        x = states
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x
    
class SoftQAgent(object):
    def __init__(self, n_state_dims, n_a_actions, hidden_units_list, lr = 3e-4, 
                 batch_size = 32, buffer_size = 10000, temp = 4, soft = True, gamma = 0.99, update_steps = 4, clip_grads = False):
        self.n_state_dims= n_state_dims
        self.n_a_actions = n_a_actions
        self.batch_size = batch_size
        self.temp = temp #smaller temp = more deterministic, larger temp = more exploration
        self.soft = soft
        self.gamma = gamma
        
        self.train_q = QModel(n_input_dims = n_state_dims, n_actions = self.n_a_actions, hidden_units_list = hidden_units_list)
        self.target_q = QModel(n_input_dims = n_state_dims, n_actions = self.n_a_actions, hidden_units_list = hidden_units_list)
        self.target_q.load_state_dict(self.train_q.state_dict())
        
        self.optimizer = torch.optim.Adam(self.train_q.parameters(), lr = lr)
        self.experience_buffer = ExperienceBuffer(buffer_size)
        self.learn_steps = 0
        self.update_steps = update_steps
        self.clip_grads = clip_grads

    def new_episode(self):
        pass

    def set_temp(self, temp):
        self.temp = temp
    
    def predict_batch(self, state):
        with torch.no_grad():
            q = self.train_q(state)
            v = self.getV(q)
            dist = torch.exp((q-v)/self.temp)
            probs = dist / torch.sum(dist, dim=1, keepdim=True)
            try:
                m = Categorical(probs)
            except ValueError:
                print('probs:\n{}\n'.format(probs))
                print('probs max:\n{}\n'.format(probs.max()))
                print('probs min:\n{}\n'.format(probs.min()))
                print('probs sums:\n{}\n'.format(probs.sum(1)))
                print('probs sums max:\n{}\n'.format(probs.sum(1).max()))
                print('probs sums min:\n{}\n'.format(probs.sum(1).min()))

                print('dist:\n{}\n'.format(dist))
                print('dist max:\n{}\n'.format(dist.max()))
                print('dist min:\n{}\n'.format(dist.min()))
                print('dist sums:\n{}\n'.format(dist.sum(1)))
                print('dist sums max:\n{}\n'.format(dist.sum(1).max()))
                print('dist sums min:\n{}\n'.format(dist.sum(1).min()))
                raise
        return m

    def predict(self, state):
        if len(state.shape) == 2:
            return self.predict_batch(state)
        elif len(state.shape) > 2:
            raise ValueError
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.train_q(state)
            v = self.getV(q).squeeze()
            dist = torch.exp((q-v)/self.temp)
            dist = dist / torch.sum(dist)
            dist_values = dist.detach().numpy()[0]
            m = Categorical(dist)
        return m
        
    def getV(self, q_value):
        v = self.temp * torch.log(torch.sum(torch.exp(q_value/self.temp), dim=1, keepdim=True))
        return v
    
    def act(self, state, eps = 0):
        if self.soft:
            return self.act_soft(state)
        else:
            return self.act_egreedy(state, eps = eps)
    
    def act_egreedy(self, state, eps = 0):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.train_q(state)

        q = q.detach().numpy()
        if random.random() > eps:
            a = np.argmax(q) 
        else:
            a = random.choice(np.arange(self.n_a_actions))
        return None, a
    
    def act_soft(self, state):
        m = self.predict(state)
        a = m.sample()
        return m, a
    
    def add_experience(self, e):
        self.experience_buffer.add(e)

    def add_experiences(self, es):
        for e in es:
            self.experience_buffer.add(e)

    def batch_add_experience(self, a_state, a_actions, r_a, next_a_state = None, done = True):
        batch_size = len(a_state)
        if next_a_state is None:
            next_a_state = [torch.zeros_like(a_state[0])]*batch_size
        dones_l = [done] * batch_size

        list_vals = [a_state, a_actions, r_a, next_a_state, dones_l]
        es = list(zip(*list_vals))
        self.add_experiences(es)

    def train(self):
        loss = 0
        if self.experience_buffer.get_size() >= self.batch_size:
            self.learn_steps += 1
            if self.learn_steps % self.update_steps == 0:
                self.update_target()
            
            states, actions, rewards, next_states, terminates = self.experience_buffer.sample(self.batch_size)

            with torch.no_grad():
                next_q = self.target_q(next_states)
                next_v = self.getV(next_q)
                y = rewards + (1 - terminates) * self.gamma * next_v

            loss = nn.functional.mse_loss(self.train_q(states).gather(1, actions), y)
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.train_q.parameters(), 4.0)
            self.optimizer.step()
        return loss
    
    def update_target(self):
        self.target_q.load_state_dict(self.train_q.state_dict())
    
    def save_model(self, file_prefix):
        train_q_file = file_prefix + 'train_q.pt'
        target_q_file = file_prefix + 'target_q.pt'
        torch.save(self.train_q.state_dict(), train_q_file)
        torch.save(self.target_q.state_dict(), target_q_file)
    
    def load_model(self, file_prefix):
        train_q_file = file_prefix + 'train_q.pt'
        target_q_file = file_prefix + 'target_q.pt'
        self.train_q.load_state_dict(torch.load(train_q_file))
        self.target_q.load_state_dict(torch.load(target_q_file))
        