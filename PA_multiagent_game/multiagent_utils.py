import sys
import glob

sys.path.insert(0, '..')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import torch
from torch.distributions import Categorical
from IPython import display

from agents.soft_q import SoftQAgent
from multi_channel_RI import MCCPERDPAgent

######### General #######################################################

def smooth_plot(values, window = 100):
    assert window >= 1
    n = len(values)
    if n < 2:
        return values
    elif n < window:
        window = int(np.floor(n/2))
    else:
        window = int(window)

    cs_values = np.cumsum(values)
    smooth_values = (cs_values[window:] - cs_values[:-window]) / window
    smooth_xs = np.arange(len(smooth_values)) + (window/2)
    return smooth_xs, smooth_values

##### Training Function ##########################################################
def train(principal_pol, agent_pol, env, n_iters=500, hist=None, train_principal=True, train_agent=True,
          normalize_t=False, normalize_n_a=False, plot = True, **kwargs):
    #train_principal and train_agent arguments used if want to stagger training
    
    assert isinstance(principal_pol, MCCPERDPAgent)

    if isinstance(agent_pol, SoftQAgent):
        agent_arch_type = 'SQA'
    elif isinstance(agent_pol, MCCPERDPAgent):
        agent_arch_type = 'RIA'
    else:
        raise NotImplementedError("Agent type not implemented")

    n_agents = env.n_agents
#     horizon = env.horizon

    # only add things to history if we are training both the principal and agent
    if train_principal and train_agent:
        if hist is None:
            hist = {'r_a': [], 'r_p': [], 'ext_r_a': [], 'ext_r_p': [], 'mi': [], 'ep_r_a': [], 'ep_r_p':[], 'ext_ep_r_a':[], 'ext_ep_r_p':[]}
    iter_vals = range(n_iters)
    if not plot:
        iter_vals = tqdm.tqdm(iter_vals)
    for _ in iter_vals:
        p_state = env.reset()
        horizon = env.horizon


        a_states = None
        r_a = None
        a_actions = None
        done = False

        # Principal and Agent Rewards
        rs_seq_a = []
        rs_seq_p = []

        # Principal and Agent EXTRINSIC Rewards
        ext_rs_seq_a = []
        ext_rs_seq_p = []
        
        principal_pol.new_episode()
        agent_pol.new_episode()

        while not done:

            # Step the principal policy
            p_actions, total_p_mi_costs = principal_pol.act(p_state)
            next_a_states = env.principal_step(p_actions)

            # Store stuff in the agent buffer, if appropriate
            if train_agent:
                if (a_states is not None) and (agent_arch_type == 'SQA'):
                    agent_pol.batch_add_experience(a_states, a_actions, r_a,
                                                       next_a_state=next_a_states, done=False)

            a_states = next_a_states

            # Step the agent policy
            if (agent_arch_type == 'SQA'):
                _, a_actions = agent_pol.act(a_states)
                a_actions = a_actions.detach().numpy()
                total_a_mi_costs = 0
            else:
                a_actions, total_a_mi_costs = agent_pol.act(a_states)

            (r_as, r_p, r_a), p_state, done = env.agent_step(a_actions)
            #r_as is a 2d array of rewards [agent1 rewards, agent2 rewards,... agentn rewards], while r_a is one long array of length batch_size * n_agents. r_a = np.concatenate(r_as) and r_as = r_a.reshape(n_agents, batch_size).T

            ext_r_a = np.array(r_a)
            ext_r_p = np.array(r_p)

            # Add mi costs
            r_a -= total_a_mi_costs
            r_p -= total_p_mi_costs
            
            #Normalize if applicable
            if normalize_t:
                r_a = r_a / env.horizon
                r_p = r_p / env.horizon
                ext_r_a = ext_r_a / env.horizon
                ext_r_p = ext_r_p / env.horizon
            if normalize_n_a:
                r_p = r_p / float(n_agents)
                ext_r_p = ext_r_p / float(n_agents)

            # Accumulate rewards
            rs_seq_a.append(r_a)
            rs_seq_p.append(r_p)

            ext_rs_seq_a.append(ext_r_a)
            ext_rs_seq_p.append(ext_r_p)


        # The game just ended, so we need to...

        #### TRAIN AGENT ####
        if train_agent:
            if agent_arch_type == 'SQA':
                agent_pol.batch_add_experience(a_states, a_actions, r_a,
                                                   next_a_state=None,
                                                   done=True)
                    
                _ = agent_pol.train()
            else:
                _ = agent_pol.end_episode(rs_seq_a)
                
        #### TRAIN PRINCIPAL ####
        if train_principal:
            _ = principal_pol.end_episode(rs_seq_p)
            
        # Log things for visualization
        if train_principal and train_agent:
            
            avg_rs_a = np.stack(rs_seq_a).mean(1)
            hist['r_a'].append(avg_rs_a)
            
            
            avg_rs_p = np.stack(rs_seq_p).mean(1)
            hist['r_p'].append(avg_rs_p)

            avg_ext_rs_a = np.stack(ext_rs_seq_a).mean(1)
            hist['ext_r_a'].append(avg_ext_rs_a)

            avg_ext_rs_p = np.stack(ext_rs_seq_p).mean(1)
            hist['ext_r_p'].append(avg_ext_rs_p)
            
            hist['ep_r_a'].append(np.sum(avg_rs_a ))
            hist['ep_r_p'].append(np.sum(avg_rs_p))
            hist['ext_ep_r_a'].append(np.sum(avg_ext_rs_a))
            hist['ext_ep_r_p'].append(np.sum(avg_ext_rs_p))


            channel_mis = principal_pol.get_mis_channels()
            for channel_name, mi_val in channel_mis:
                if channel_name not in hist:
                    hist[channel_name] = {}
                if env.horizon not in hist[channel_name]:
                    hist[channel_name][env.horizon] = []
                hist[channel_name][env.horizon].append(mi_val)
    return hist



##### Plotting the History ##########################################################
def plot_hist_signaling_vary_h(hist, axes=None, plot_smoothed_only = False):
    matplotlib.rcParams['image.aspect'] = 'auto'
    matplotlib.rcParams['image.interpolation'] = 'none'

    if axes is None:
        _, axes = plt.subplots(2, 4, figsize=(16, 8))

    (ax0, ax1, ax2, ax3) = axes[0]
    (ax4, ax5, ax6, ax7) = axes[1]
    for subax in axes:
        for ax in subax:
            ax.cla()

    total_ra = hist['ep_r_a']
    total_rp = hist['ep_r_p']
    total_ext_ra = hist['ext_ep_r_a']
    total_ext_rp = hist['ext_ep_r_p']

    if not plot_smoothed_only:
        ax0.plot(total_ra, color='b', alpha=0.2)
        ax0.plot(total_ext_ra, color='r', alpha=0.2)
    ax0.plot(*smooth_plot(total_ra, window=100), color='b')
    ax0.plot(*smooth_plot(total_ext_ra, window=100), color='r')
    ax0.grid(b=True)

    if not plot_smoothed_only:
        ax4.plot(total_rp, color='b', alpha=0.2)
        ax4.plot(total_ext_rp, color='r', alpha=0.2)
    ax4.plot(*smooth_plot(total_rp, window=100), color='b')
    ax4.plot(*smooth_plot(total_ext_rp, window=100), color='r')
    ax4.grid(b=True)
    
    max_h = max(hist['mi-last_effort'].keys())
    min_h = min(hist['mi-last_effort'].keys())
    
    ax1.imshow(np.array(hist['mi-last_effort'][min_h]), vmin=0, vmax=2.5)
    ax2.imshow(np.array(hist['mi-last_individual_outputs'][min_h]), vmin=0, vmax=2.5)
    ax3.imshow(np.array(hist['mi-last_wage_hours_output_time'][min_h]), vmin=0, vmax=2.5)

    ax0.set_title('Agent Reward')
    ax4.set_title('Principal Reward (includes MI cost)')
    
    ax1.set_title(f'MI: Effort {min_h}')
    ax2.set_title(f'MI: Individual Outputs {min_h}')
    ax3.set_title(f'MI: Others {min_h}')
    
    ax5.imshow(np.array(hist['mi-last_effort'][max_h]), vmin=0, vmax=2.5)
    ax6.imshow(np.array(hist['mi-last_individual_outputs'][max_h]), vmin=0, vmax=2.5)
    ax7.imshow(np.array(hist['mi-last_wage_hours_output_time'][max_h]), vmin=0, vmax=2.5)
    ax5.set_title(f'MI: Effort {max_h}')
    ax6.set_title(f'MI: Individual Outputs {max_h}')
    ax7.set_title(f'MI: Others {max_h}')

# ###### Function for naming savefiles #########################################
def get_savestr_allh(folder, principal_effort_mi_cost, principal_output_mi_cost, normalize_t, *args, **kwargs):
    effort_name = f'mipe{principal_effort_mi_cost:.2f}'
    output_name = f'mipe{principal_output_mi_cost:.2f}'
    normalize_name = f'nt{int(normalize_t)}'

    return f'{folder}/{effort_name}_{normalize_name}_{output_name}'


