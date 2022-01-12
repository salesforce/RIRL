#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import torch
import numpy as np

class PAMultiagentEnv(object):
    def __init__(self,
                 agent_type_dist = [0.5, 0.5],
                 agent_base_skill = [0.5, 1],
                 agent_hrs_cost_mult = 0.2,
                 principal_profit_mult = 1.5,
                 horizon = 2,
                 batch_size = 32,
                 n_agents = 2,
                 agent_arch_type = 'SQA', *args, **kwargs
                 ):
        self.agent_type_dist = agent_type_dist
        self.agent_base_skill = agent_base_skill
        self.agent_hrs_cost_mult = agent_hrs_cost_mult
        self.principal_profit_mult = principal_profit_mult
        self.horizon = horizon
        self.agent_arch_type = agent_arch_type
        
        self.n_agents = int(n_agents)
        assert self.n_agents > 0

        self.batch_size = int(batch_size)
        assert self.batch_size > 0

        # These are overwritten in self.reset()
        self.current_base_skill = np.zeros((self.batch_size, self.n_agents))
        self.agent_type = np.zeros((self.batch_size, self.n_agents), dtype=int)
        self.last_wage = np.zeros((self.batch_size, self.n_agents))
        self.last_hour = np.zeros((self.batch_size, self.n_agents))
        self.last_output_cumulative = np.zeros(self.batch_size)
        self.last_output_individual = np.zeros((self.batch_size, self.n_agents))

        self.t = np.zeros(self.batch_size, dtype=np.int)
        self.wage = np.zeros((self.batch_size, self.n_agents))

#         self.reset()

    
    def get_agent_state(self):
        #returns agent state stacked together of size(batch_size * n_agents, 3) 
        
        agent_states = np.array([np.concatenate(self.agent_type.T), np.concatenate(self.wage.T), np.tile(self.t, self.n_agents)]).T
        agent_states = torch.as_tensor(agent_states, dtype=torch.float32)
        assert agent_states.shape[0] == self.batch_size * self.n_agents
        if self.agent_arch_type == 'SQA':
            return agent_states
        else:
            return {
            'state': agent_states
        }

    def get_principal_state(self):
        # Output formatted for multi-channel RI policy
        return {
            'last_individual_outputs': torch.as_tensor(self.last_output_individual, dtype=torch.float32),
            'last_wage_hours_output_time': torch.as_tensor(
                np.hstack([
                    self.last_wage,
                    self.last_hour,
                    self.last_output_cumulative.reshape(-1, 1),
                    self.t.reshape(-1, 1)
                ]),
                dtype=torch.float32
            )
        }
        
    def reset(self, agent_type = None, horizon = None):
        if agent_type is None:
            self.agent_type = np.random.choice(
                a=list(range(len(self.agent_type_dist))),
                size=(self.batch_size, self.n_agents),
                p=self.agent_type_dist
            )
        else:
            agent_type = np.array(agent_type).astype(np.int)
            nb, na = agent_type.shape
            assert nb == self.batch_size
            assert na == self.n_agents
            assert np.max(agent_type) < len(self.agent_type_dist)
            assert np.min(agent_type) >= 0
            self.agent_type = agent_type

        self.current_base_skill = np.array(
            [self.agent_base_skill[t] for t in self.agent_type]
        )
       
        self.last_wage = np.zeros((self.batch_size, self.n_agents))
        self.last_hour = np.zeros((self.batch_size, self.n_agents))
        self.last_output_cumulative = np.zeros(self.batch_size)
        self.last_output_individual = np.zeros((self.batch_size, self.n_agents))

        self.t = np.zeros(self.batch_size, dtype=np.int)
        self.wage = np.zeros((self.batch_size, self.n_agents))

        return self.get_principal_state()
    
    def principal_step(self, principal_action):
        assert len(principal_action) == self.batch_size

        # Principal actions update the wage
        self.wage = np.array(principal_action, dtype=np.int)
        if self.n_agents == 1:
            self.wage = self.wage.reshape(-1, 1)

        # Agent is next to act. Output its state.
        return self.get_agent_state()
    
#     def agent_utility(self, hrs_action, agent_i):

#         pay_util = hrs_action * self.wage[:, agent_i]
#         labor_cost = (hrs_action ** 2) * (self.agent_hrs_cost_mult)

#         return pay_util - labor_cost
    
    def agent_utility(self, hrs_action):

        pay_util = hrs_action * np.concatenate(self.wage.T)
        labor_cost = (hrs_action ** 2) * (self.agent_hrs_cost_mult)

        return pay_util - labor_cost

    def principal_utility(self, output_cumulative, hrs_action):
        revenue = output_cumulative * self.principal_profit_mult
        cost  = (self.wage * hrs_action).sum(1)
        return revenue - cost
    
    def agent_step(self, agent_actions):
        hrs_action = agent_actions.reshape(self.n_agents, self.batch_size).T

#         # Look up the (hours, effort) action indicated by these actions
#         hour_effort_array = np.array([self.idx_to_action[a] for a in agent_action])
#         hrs_action = hour_effort_array[:, 0]
#         effort_action = hour_effort_array[:, 1]
        
        #total skill is base skill + effort increase
#         total_skill = self.current_base_skill + effort_action
        
        #output is total_effort * hrs
        #individual_outputs
        output = self.current_base_skill * hrs_action
        output_cumulative = output.sum(1)

        # We can now calculate reward for this pair of (principal, agent) actions
        
        #get reward for each agent
        r_as = []
#         for agent_i in range(self.n_agents):
#             r_as.append(self.agent_utility(hrs_action[:, agent_i], agent_i))
        
        #stack rewards
#         r_a = np.concatenate(r_as)
        r_a = self.agent_utility(agent_actions)
        r_a_shape = r_a.shape
        assert len(r_a_shape) == 1
        assert r_a_shape[0] == self.batch_size * self.n_agents
        r_p = self.principal_utility(output_cumulative, hrs_action)

        # Update our state trackers
        self.last_wage = np.array(self.wage)
        self.last_hour = hrs_action
        self.last_output_individual = output
        self.last_output_cumulative = output_cumulative
        self.t += 1

        done = self.t >= self.horizon
        if np.any(done):
            assert np.all(done)
            done = True
        else:
            done = False

        # Output rewards tuple, principal state, and done flag
        #r_as is a 2d array of rewards [agent1 rewards, agent2 rewards,... agentn rewards], while r_a is one long array of length batch_size * n_agents. r_a = np.concatenate(r_as) and r_as = r_a.reshape(n_agents, batch_size).T
        return (r_as, r_p, r_a), self.get_principal_state(), done


class PAMultiagentSignalingEnv(PAMultiagentEnv):
    def __init__(self,
                 *args,
                 agent_effort_cost_mult=[1, 0.5],
                 n_agent_hour_actions=8,
                 n_agent_effort_actions=3,
                 agent_effort_increment = 1,
                 **kwargs):

        self.agent_effort_cost_mult = np.array(agent_effort_cost_mult)
        super().__init__(*args, **kwargs)
        assert len(self.agent_effort_cost_mult) == len(self.agent_type_dist)

        self.current_effort_cost_mult = np.zeros(self.batch_size)

        self.n_agent_hour_actions = int(n_agent_hour_actions)
        assert self.n_agent_hour_actions >= 2

        self.n_agent_effort_actions = int(n_agent_effort_actions)
        assert self.n_agent_effort_actions >= 1

        self.last_effort = np.zeros_like(self.last_hour)
        self.agent_effort_increment = agent_effort_increment

        # Assign each agent (hrs, effort) action to a unique integer
        self.idx_to_action = {}
        idx = 0
        for hr in range(self.n_agent_hour_actions):
            for e in range(self.n_agent_effort_actions):
                self.idx_to_action[idx] = (hr, e)
                idx += 1

#         self.reset()

    def reset(self, agent_type=None, horizon = None):
        if agent_type is None:
            self.agent_type = np.random.choice(
                a=list(range(len(self.agent_type_dist))),
                size=(self.batch_size, self.n_agents),
                p=self.agent_type_dist
            )
        else:
            agent_type = np.array(agent_type).astype(np.int)
            nb, na = agent_type.shape
            assert nb == self.batch_size
            assert na == self.n_agents
            assert np.max(agent_type) < len(self.agent_type_dist)
            assert np.min(agent_type) >= 0
            self.agent_type = agent_type

        self.current_base_skill = np.array(
            [self.agent_base_skill[t] for t in self.agent_type]
        )

        self.current_base_skill = np.array(
            [self.agent_base_skill[t] for t in self.agent_type]
        )
        self.current_effort_cost_mult = np.array(
            [self.agent_effort_cost_mult[t] for t in self.agent_type]
        )

        self.last_wage = np.zeros((self.batch_size, self.n_agents))
        self.last_hour = np.zeros((self.batch_size, self.n_agents))
        self.last_effort = np.zeros((self.batch_size, self.n_agents))
        self.last_output_cumulative = np.zeros(self.batch_size)
        self.last_output_individual = np.zeros((self.batch_size, self.n_agents))

        self.t = np.zeros(self.batch_size, dtype=np.int)
        self.wage = np.zeros((self.batch_size, self.n_agents))

        return self.get_principal_state()

    def agent_step(self, agent_actions):
        # hrs_action = agent_actions.reshape(self.n_agents, self.batch_size).T

        # Look up the (hours, effort) action indicated by these actions
        he_array = np.array([self.idx_to_action[a] for a in agent_actions])
        hrs_action = he_array[:, 0].reshape(self.n_agents, self.batch_size).T
        effort_action = he_array[:, 1].reshape(self.n_agents, self.batch_size).T

        # total skill is base skill + effort increase
        # total_skill = self.current_base_skill + effort_action
        total_skill = self.current_base_skill + (effort_action * self.agent_effort_increment)

        # output is total_effort * hrs
        # individual_outputs
        output = total_skill * hrs_action
        output_cumulative = output.sum(1)

        # We can now calculate reward for this pair of (principal, agent) actions

        # get reward for each agent
        r_as = []
        for agent_i in range(self.n_agents):
            r_as.append(
                self.agent_utility(
                    hrs_action[:, agent_i],
                    effort_action[:, agent_i],
                    agent_i
                )
            )

        # stack rewards
        r_a = np.concatenate(r_as)
        r_a_shape = r_a.shape
        assert len(r_a_shape) == 1
        assert r_a_shape[0] == self.batch_size * self.n_agents
        r_p = self.principal_utility(output_cumulative, hrs_action)

        # Update our state trackers
        self.last_wage = np.array(self.wage)
        self.last_hour = hrs_action
        self.last_effort = effort_action
        self.last_output_individual = output
        self.last_output_cumulative = output_cumulative
        self.t += 1

        done = self.t >= self.horizon
        if np.any(done):
            assert np.all(done)
            done = True
        else:
            done = False

        # Output rewards tuple, principal state, and done flag
        # r_as is a 2d array of rewards [agent1 rewards, agent2 rewards,... agentn rewards], while r_a is one long array of length batch_size * n_agents. r_a = np.concatenate(r_as) and r_as = r_a.reshape(n_agents, batch_size).T
        return (r_as, r_p, r_a), self.get_principal_state(), done

    def agent_utility(self, hrs_action, effort_action, agent_i):
        pay_util = hrs_action * self.wage[:, agent_i]
        effort_cost_scale = (self.current_effort_cost_mult[:, agent_i] *
                             effort_action) + 1
        labor_cost = (hrs_action ** 2) * (self.agent_hrs_cost_mult * effort_cost_scale)

        return pay_util - labor_cost

    def get_principal_state(self):
        # Output formatted for multi-channel RI policy
        return {
            'last_effort': torch.as_tensor(self.last_effort, dtype=torch.float32),
            'last_individual_outputs': torch.as_tensor(self.last_output_individual, dtype=torch.float32),
            'last_wage_hours_output_time': torch.as_tensor(
                np.hstack([
                    self.last_wage,
                    self.last_hour,
                    self.last_output_cumulative.reshape(-1, 1),
                    self.t.reshape(-1, 1)
                ]),
                dtype=torch.float32
            )
        }

class PAMultiagentSignalingEnvVaryH(PAMultiagentSignalingEnv):
    def __init__(self,
                 *args,
                 horizons = list(range(2, 11)),
                 **kwargs):

        self.horizons = horizons
        super().__init__(*args, **kwargs)

    def reset(self, agent_type=None, horizon = None):
        if agent_type is None:
            self.agent_type = np.random.choice(
                a=list(range(len(self.agent_type_dist))),
                size=(self.batch_size, self.n_agents),
                p=self.agent_type_dist
            )
        else:
            agent_type = np.array(agent_type).astype(np.int)
            nb, na = agent_type.shape
            assert nb == self.batch_size
            assert na == self.n_agents
            assert np.max(agent_type) < len(self.agent_type_dist)
            assert np.min(agent_type) >= 0
            self.agent_type = agent_type
        if horizon is None:
            self.horizon = np.random.choice(self.horizons)
        else:
            self.horizon = horizon

        self.current_base_skill = np.array(
            [self.agent_base_skill[t] for t in self.agent_type]
        )

        self.current_base_skill = np.array(
            [self.agent_base_skill[t] for t in self.agent_type]
        )
        self.current_effort_cost_mult = np.array(
            [self.agent_effort_cost_mult[t] for t in self.agent_type]
        )

        self.last_wage = np.zeros((self.batch_size, self.n_agents))
        self.last_hour = np.zeros((self.batch_size, self.n_agents))
        self.last_effort = np.zeros((self.batch_size, self.n_agents))
        self.last_output_cumulative = np.zeros(self.batch_size)
        self.last_output_individual = np.zeros((self.batch_size, self.n_agents))

        self.t = np.zeros(self.batch_size, dtype=np.int)
        self.wage = np.zeros((self.batch_size, self.n_agents))

        return self.get_principal_state()

    def agent_step(self, agent_actions):
        # hrs_action = agent_actions.reshape(self.n_agents, self.batch_size).T

        # Look up the (hours, effort) action indicated by these actions
        he_array = np.array([self.idx_to_action[a] for a in agent_actions])
        hrs_action = he_array[:, 0].reshape(self.n_agents, self.batch_size).T
        effort_action = he_array[:, 1].reshape(self.n_agents, self.batch_size).T

        # total skill is base skill + effort increase
        # total_skill = self.current_base_skill + effort_action
        total_skill = self.current_base_skill + (effort_action * self.agent_effort_increment)

        # output is total_effort * hrs
        # individual_outputs
        output = total_skill * hrs_action
        output_cumulative = output.sum(1)

        # We can now calculate reward for this pair of (principal, agent) actions

        # get reward for each agent
        r_as = []
        for agent_i in range(self.n_agents):
            r_as.append(
                self.agent_utility(
                    hrs_action[:, agent_i],
                    effort_action[:, agent_i],
                    agent_i
                )
            )

        # stack rewards
        r_a = np.concatenate(r_as)
        r_a_shape = r_a.shape
        assert len(r_a_shape) == 1
        assert r_a_shape[0] == self.batch_size * self.n_agents
        r_p = self.principal_utility(output_cumulative, hrs_action)

        # Update our state trackers
        self.last_wage = np.array(self.wage)
        self.last_hour = hrs_action
        self.last_effort = effort_action
        self.last_output_individual = output
        self.last_output_cumulative = output_cumulative
        self.t += 1

        done = self.t >= self.horizon
        if np.any(done):
            assert np.all(done)
            done = True
        else:
            done = False

        # Output rewards tuple, principal state, and done flag
        # r_as is a 2d array of rewards [agent1 rewards, agent2 rewards,... agentn rewards], while r_a is one long array of length batch_size * n_agents. r_a = np.concatenate(r_as) and r_as = r_a.reshape(n_agents, batch_size).T
        return (r_as, r_p, r_a), self.get_principal_state(), done

    def agent_utility(self, hrs_action, effort_action, agent_i):
        pay_util = hrs_action * self.wage[:, agent_i]
        effort_cost_scale = (self.current_effort_cost_mult[:, agent_i] *
                             effort_action) + 1
        labor_cost = (hrs_action ** 2) * (self.agent_hrs_cost_mult * effort_cost_scale)

        return pay_util - labor_cost

    def get_principal_state(self):
        # Output formatted for multi-channel RI policy
        return {
            'last_effort': torch.as_tensor(self.last_effort, dtype=torch.float32),
            'last_individual_outputs': torch.as_tensor(self.last_output_individual, dtype=torch.float32),
            'last_wage_hours_output_time': torch.as_tensor(
                np.hstack([
                    self.last_wage,
                    self.last_hour,
                    self.last_output_cumulative.reshape(-1, 1),
                    self.t.reshape(-1, 1),
                    self.horizon * np.ones((self.batch_size, 1))
                ]),
                dtype=torch.float32
            )
        }
    
    def get_agent_state(self):
        #returns agent state stacked together of size(batch_size * n_agents, 3) 
        
        agent_states = np.array([np.concatenate(self.agent_type.T), np.concatenate(self.wage.T), np.tile(self.t, self.n_agents), self.horizon * np.ones(self.batch_size * self.n_agents)]).T
        agent_states = torch.as_tensor(agent_states, dtype=torch.float32)
        assert agent_states.shape[0] == self.batch_size * self.n_agents
        if self.agent_arch_type == 'SQA':
            return agent_states
        else:
            return {
            'state': agent_states
        }
    