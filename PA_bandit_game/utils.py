import random
import pickle
import sys
import tqdm
sys.path.insert(0,'..')
sys.path.insert(0,'../agents')

import numpy as np
import torch
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
from IPython import display

from agents.mi_classifier import MIClassifier
from agents.pg_bandit_mvg_model import PGBanditMVGModel, PGBanditLNModel

"""Agent noise model."""
def get_a_o_mat_exact_nc(
        n_a_actions, n_a_outputs, random_factor = 2.0, *args, **kwargs
):
    """
    The exact action to output probability matrix for exponentially decaying noise

    Args:
         n_a_actions (int): number of agent actions
         n_a_outputs (int): number of agent outputs that can result from the actions
         random_factor (float): parameter controlling the shape of the decay
    """
    a_o_diff = int((n_a_outputs - n_a_actions)/2.0)
    a_o_mat = np.zeros((n_a_actions, n_a_outputs))
    for a in range(n_a_actions):
        a_idx = a+a_o_diff
        for o in range(n_a_outputs):
            diff = np.abs(a_idx - o)
            if diff == 0:
                a_o_mat[a, o] = 0
            else:
                a_o_mat[a, o] = random_factor**diff
        a_o_mat[a, :] = a_o_mat[a, :]/np.sum(a_o_mat[a])*0.3

        a_o_mat[a, a+a_o_diff] = 0.7
    return a_o_mat

"""Rewards."""
def get_principal_reward_stochastic_schedule(
        pay_schedule_distribution,
        n_a_outputs,
        n_a_actions,
        principal_profit_multiplier,
        agent_work_multiplier,
        agent_pay_multiplier,
        agent_exp,
        qtemp_agent,
        a_o_mat,
        n_samples = 100,
        **kwargs
):
    """
    Estimates expected principal reward (profit) given a pay schedule
    """

    means, stds = pay_schedule_distribution
    m = Normal(means, stds)

    agent_pay_util_per_output_samples = []
    agent_pay_per_action_samples = []
    pay_schedules = []

    for _ in range(n_samples):
        pay = torch.clamp(m.sample(), min = 0)

        pay = pay.detach().numpy()
        agent_pay_util_per_output = ((pay*agent_pay_multiplier + 1.0)**(1.0-agent_exp)-1.0)/(1.0-agent_exp)
        agent_pay_util_per_output_samples.append(agent_pay_util_per_output)
        agent_pay_per_action_samples.append(pay.dot(a_o_mat.T))
        pay_schedules.append(pay)

    agent_pay_per_action_averaged = np.mean(
        agent_pay_per_action_samples, axis = 0
    )
    agent_pay_util_per_output_averaged = np.mean(
        agent_pay_util_per_output_samples, axis = 0
    )
    agent_pay_util_per_actions = agent_pay_util_per_output_averaged.dot(a_o_mat.T)

    agent_work_costs = ((np.arange(n_a_actions)/(n_a_actions-1)))*agent_work_multiplier
    agent_util_per_action = agent_pay_util_per_actions - agent_work_costs.reshape(1, -1)

    m = Categorical(logits = qtemp_agent * torch.FloatTensor(agent_util_per_action))
    action_probability = m.probs.detach().numpy()

    output_probability = action_probability.dot(a_o_mat)

    pay_schedules = np.mean(np.array(pay_schedules), axis = 0)
    principal_gains = np.arange(n_a_outputs)/(n_a_outputs - 1) * principal_profit_multiplier

    principal_gain_per_output = principal_gains.reshape(1, -1) - pay_schedules
    principal_reward = np.sum(output_probability * principal_gain_per_output, axis = 1)
    agent_utility = np.sum(action_probability * agent_util_per_action, axis = 1)

    return (
        principal_reward,
        agent_utility,
        (action_probability, output_probability,
         agent_util_per_action, agent_pay_util_per_output_averaged,
         pay_schedules, agent_pay_per_action_averaged)
    )

def get_principal_reward_deterministic_schedule(
        pay,
        n_a_outputs,
        n_a_actions,
        principal_profit_multiplier,
        agent_work_multiplier,
        agent_pay_multiplier,
        agent_exp,
        qtemp_agent,
        a_o_mat,
        **kwargs
):
    """For estimating rewards; only used here for plotting."""

    agent_util_per_output = ((pay*agent_pay_multiplier + 1.0)**(1.0-agent_exp)-1.0)/(1.0-agent_exp)
    principal_gain_per_output = np.arange(n_a_outputs)/(n_a_outputs - 1) * principal_profit_multiplier - pay
    average_pay_per_action = a_o_mat.dot(pay)

    agent_utilities = agent_util_per_output.dot(a_o_mat)
    agent_work_costs = ((np.arange(n_a_actions)*(1/(n_a_actions-1)))**1.0)*agent_work_multiplier

    agent_utilities = agent_utilities - agent_work_costs

    m = Categorical(logits = qtemp_agent * torch.FloatTensor(agent_utilities))
    output_probability = m.probs.numpy().dot(a_o_mat)
    principal_reward = output_probability.dot(principal_gain_per_output)
    agent_utility = m.probs.numpy().dot(agent_utilities)
    agent_pay = m.probs.numpy().dot(average_pay_per_action)

    return (
        principal_reward,
        (m.probs.numpy(), output_probability, agent_utilities,
         agent_utility, average_pay_per_action, agent_pay)
    )

"""Mutual Information estimation / learning (helper functions)."""
def get_o_a_a_p(n_a_outputs, output_probs, m, n_samples):
    all_outputs = np.arange(n_a_outputs)
    outputs = np.random.choice(all_outputs, p = output_probs, size = n_samples)
    actions_all = m.sample()
    x = torch.LongTensor(outputs).view(-1, 1)
    a_p = torch.gather(actions_all, 1, x)
    o_a = torch.FloatTensor(outputs)/float(n_a_outputs - 1)
    a_p = a_p.squeeze()
    return o_a, a_p

def get_xy_target(o_a, a_p):
    targets = torch.cat(
        [torch.ones(o_a.shape[0], dtype=torch.long),
         torch.zeros(o_a.shape[0],dtype=torch.long)],
        dim=0
    )
    xy_from_policy = torch.stack([o_a, a_p], dim=1)
    xy_from_marginals = torch.stack([o_a, a_p[torch.randperm(a_p.shape[0])]], dim=1)
    xy = torch.cat([xy_from_policy, xy_from_marginals], dim=0).float()
    return xy, targets

def get_model_mi(n_samples, pay_schedule):
    mean, std = pay_schedule
    all_mean = torch.FloatTensor(mean).repeat((n_samples, 1))
    all_std = torch.FloatTensor(std).repeat((n_samples, 1))
    m = Normal(all_mean, all_std)
    return m

def update_mi_classifier(mi_classifier, xy, targets):
    mi_classifier.optimizer.zero_grad()
    loss = mi_classifier.get_loss(xy, targets)
    loss.backward()
    mi_classifier.optimizer.step()
    return loss

def get_log_odds(mi_classifier, o_a, a_p):
    xy = torch.stack([o_a, a_p], dim=1).float()
    log_odds_ratio = float(mi_classifier.log_odds_ratio(xy).detach().mean())
    return log_odds_ratio

def pretrain_mi_classifier(
        pi_p, a_o_mat, n_batch, params_dict,
        mi_classifier=None, n_iters=20, n_samples=1024
):
    if mi_classifier is None:
        mi_classifier = MIClassifier()

    mean, std, a_log_probs_p = pi_p.sample(n_batch)
    
    principal_reward, agent_utility, (action_probability, output_probability, agent_util_per_action, agent_pay_util_per_output_averaged, pay_schedules, agent_pay_per_action_averaged) \
    = get_principal_reward_stochastic_schedule(pay_schedule_distribution = (mean, std), a_o_mat = a_o_mat, n_samples = 1000, **params_dict)

    for output_probs, mean_vals, std_vals in zip(output_probability, mean, std):
        calc_mi_schedule(output_probs, (mean_vals, std_vals), n_iters = n_iters, n_samples = n_samples, mi_classifier = mi_classifier, **params_dict)
    return mi_classifier

def calc_mi_schedule(
        output_probs, pay_schedule, n_a_outputs,
        n_iters=300, n_samples=1024, mi_classifier=None, update_prob=1, **kwargs
):
    
    m = get_model_mi(n_samples, pay_schedule)
    output_probs_sum = np.sum(output_probs)
    output_probs = output_probs/output_probs_sum
    
    if mi_classifier is None:
        mi_classifier = MIClassifier()
    
    if random.random() < update_prob:
        for i in range(n_iters):
            o_a, a_p = get_o_a_a_p(n_a_outputs, output_probs, m, n_samples)
            xy, targets = get_xy_target(o_a, a_p)
            _ = update_mi_classifier(mi_classifier, xy, targets)
            
    o_a, a_p = get_o_a_a_p(n_a_outputs, output_probs, m, n_samples)
    lor = get_log_odds(mi_classifier, o_a, a_p)
       
    return lor

def calc_mi_schedule_first_shuffle_individual(
        output_probs, pay_schedule, n_a_outputs,
        n_samples, n_samples_update, mi_classifier, **kwargs
):
    m = get_model_mi(n_samples, pay_schedule)
    output_probs_sum = np.sum(output_probs)
    output_probs = output_probs/output_probs_sum

    o_a, a_p = get_o_a_a_p(n_a_outputs, output_probs, m, n_samples)
    lor = get_log_odds(mi_classifier, o_a, a_p)
    
    o_a_update = o_a[0:n_samples_update]
    a_p_update = a_p[0:n_samples_update]
    
    targets = torch.cat(
        [torch.ones(n_samples_update, dtype=torch.long),
         torch.zeros(n_samples_update,dtype=torch.long)],
        dim=0
    )
    
    xy, targets = get_xy_target(o_a_update, a_p_update)
            
    return lor, xy, targets


"""Training (and associated visualizations)"""
def batch_game(
        pi_p,
        params_dict,
        a_o_mat,
        optim_p=None,
        n_batch=128,
        updates=1,
        mi_lambda=0.0,
        mi_classifier=None,
        history=None,
):
    """
    Learn from one or more batches of pay schedules.

    This function is meant to serve as an inner loop of a training pipeline. Calling
        it performs one or more rounds of training. Depending on the inputs provided,
        this function also adds some logging of training history.

    Args:
        pi_p (PGBanditMVGModel, PGBanditLNModel): A principal policy module,
            with learnable schedule parameters.
        params_dict (dict): Dictionary containing some structural parameters.
        a_o_mat (np array): [n_agent_actions, n_agent_outputs] probability matrix.
        optim_p (torch optimizer): Optional. Optimizer for training pi_p. If not
            supplied, an optimizer is initialized internally.
        n_batch (int): Batch size for each update. Defaults to 128.
        updates (int): Number of rounds of updates (i.e. batches) to perform.
            Defaults to 1.
        mi_lambda (float): Coefficient that multiplies any mutual information penalty
            added to the schedule reward. Defaults to 0.
        mi_classifier (MIClassifier): Optional. The discriminator module used to
            estimate mutual information. If provided, this module will be trained
            along with the principal policy module. If not provided, mutual
            information penalties will always be 0.
        history (dict): Optional. A dictionary that contains the history of various
            stats over training. If provided, the dictionary will be internally
            updated at the end of each training round.
    """

    if optim_p is None:
        optim_p = torch.optim.Adam(pi_p.parameters(), lr=0.001)
    n_a_actions = params_dict['n_a_actions']
    n_a_outputs = params_dict['n_a_outputs']
    entropy_lambda = params_dict['entropy_lambda']
    
    if history is not None:
        if len(history) == 0:
            dummy_stats_dict = characterize(
                pi_p, agent_probs=n_a_actions, output_probs=n_a_outputs
            )
            for k, v in dummy_stats_dict.items():
                history[k] = [v]
        
    for _ in range(updates):
        #sample principal actions
        mean, std, a_log_prob, m_entropy = pi_p.sample_and_entropy(n_batch)
        #mean, std, of size n_batch * n_a_outputs (ex. 128 * n_a_outputs), 
        #log_probs of size n_batch
        
        #principal_reward, agent_utility of size n_batch, 
        #output_probability, agent_pay_util_per_output_averaged, pay_schedules of size n_batch * n_a_outputs, 
        #action_probability, agent_util_per_action, agent_pay_per_action_averaged of size n_batch * n_a_actions
        principal_reward, agent_utility, (action_probability, output_probability, agent_util_per_action, agent_pay_util_per_output_averaged, pay_schedules, agent_pay_per_action_averaged) \
        = get_principal_reward_stochastic_schedule(pay_schedule_distribution = (mean, std), a_o_mat = a_o_mat, n_samples = 1000, **params_dict)

        mi_penalty = 0.0
        
        if mi_classifier is not None:
            mi_penalty = []
            xys = []
            targets = []
                
            ###########
            for output_probs, mean_vals, std_vals in zip(output_probability, mean, std):
                mi_args_dict = {
                    'output_probs': output_probs,
                    'pay_schedule': (mean_vals, std_vals),
                    'n_samples': 256,
                    'n_samples_update': 20,
                    'mi_classifier': mi_classifier,
                    'n_a_outputs': n_a_outputs
                }
                lor, xy, target = calc_mi_schedule_first_shuffle_individual(
                    **mi_args_dict
                )
                xys.append(xy)
                targets.append(target)
                mi_penalty.append(lor)
            
            ######################################################
            mi_penalty =  torch.FloatTensor(mi_penalty)
            xy_all = torch.cat(xys)
            target_all = torch.cat(targets)
            _ = update_mi_classifier(mi_classifier, xy_all, target_all)
        
        #Calculate the entropy bonus if we are looking at entropy regularization
        all_e_penalties = 0
        if entropy_lambda > 0:
            all_e_penalties = []
            for output_probs, mean_vals, std_vals in zip(output_probability, mean, std):
                weighted_e_penalty, avg_e_penalty = get_entropy((mean_vals, std_vals), output_probs)
                weighted_e_penalty = min(max(weighted_e_penalty, -10), 10)
                all_e_penalties.append(weighted_e_penalty)
            all_e_penalties = torch.FloatTensor(all_e_penalties)
        
        #mi_penalty of size n
        principal_reward = torch.FloatTensor(principal_reward) - (mi_lambda * mi_penalty) + (entropy_lambda * all_e_penalties )
        #Update Principal
        optim_p.zero_grad()
        loss_p = pi_p.get_loss(torch.FloatTensor(principal_reward - principal_reward.mean()), a_log_prob, m_entropy)
        loss_p.backward()
        optim_p.step()

        #Update history for training visualizations
        if history is not None:
            stats_dict = characterize(
                pi_p, action_probability, output_probability, agent_utility,
                principal_reward, mi_penalty, loss_p, mi_lambda, all_e_penalties
            )
            for k, v in stats_dict.items():
                history[k].append(v)
    return None

def get_entropy(pay_schedule, output_probs):
    mean, std = pay_schedule
    m = Normal(mean, std)
    entropy_each = m.entropy().numpy()
    return output_probs.dot(entropy_each), np.mean(entropy_each)

def characterize(
        pi_p,
        agent_probs,
        output_probs,
        r_a=np.array([0]),
        r_p=np.array([0]),
        mi_penalty=0.0,
        loss_p=np.array([0]),
        mi_lambda=0.0,
        e_penalty=0.0
):
    """Used to track training history."""
    try:
        agent_probs = np.mean(agent_probs, axis = 0)
        output_probs = np.mean(output_probs, axis = 0)
    except:
        agent_probs = np.mean(np.array([np.ones(agent_probs)/float(agent_probs)]), axis = 0)
        output_probs = np.mean(np.array([np.ones(output_probs)/float(output_probs)]), axis = 0)

    mean, std, a_log_prob, m_entropy = pi_p.sample_and_entropy(1)
    mean_mean, mean_std = pi_p.get_mean_mean_std()
    out = {}
    
    out['agent_action_dist'] = agent_probs
    out['output_dist'] = output_probs
    out['principal_mean'] = mean_mean
    out['principal_std'] = mean_std
    m_entropy = m_entropy.detach().numpy().squeeze()
    
    n_a_outputs = int(len(m_entropy)/2)
    out['m_entropy_mean'] = m_entropy[0:n_a_outputs].mean()
    out['m_entropy_std'] = m_entropy[n_a_outputs:].mean()

    
    out['r_a'] = float(r_a.mean())
    out['r_p'] = float(r_p.mean())

    out['loss_p'] = float(loss_p)
    try:
        out['mi_penalty'] = float(mi_penalty.mean())
    except:
        out['mi_penalty'] = mi_penalty
    out['mi_lambda'] = mi_lambda
    out['log_probs'] = float(a_log_prob.detach().mean())
    
    try:
        out['e_penalty'] = float(e_penalty.mean())
    except:
        out['e_penalty'] = e_penalty
        
    return out

def train_principal(
        pi_p,
        optim_p,
        mi_classifier,
        mi_lambda,
        n_batch,
        a_o_mat,
        params_dict,
        history=None,
        n_timesteps=100000,
        anneal_mi=True,
        u_begin=0,
        anneal_speed=2.0,
        entropy_regularization=True,
        entropy_anneal=1.0,
        entropy_begin=1.0,
        should_plot=True
):
    """
    Wraps batch_game to create more of a trianing pipeline plus visualization.

    Args:
        pi_p: See batch_game docstring.
        optim_p: ^^^
        mi_classifier: ^^^
        mi_lambda: ^^^
        n_batch: ^^^
        a_o_mat: ^^^
        params_dict: ^^^
        history: ^^^
        n_timesteps (int): The number of training rounds to perform.
        anneal_mi (bool): Flag indicating whether to anneal the mi coeffecient up to
            its final value during trianing.
        u_begin (int): Offset between the timestep used in the annealing schedule and
            the timestep of the actual training pipeline. This allows you to start
            the annealing process at some >0 value. Defaults to 0.
        anneal_speed (float): Rate at which mi coefficient anneals to its final
            value. Rate is units per 10000 timesteps. Defaults to 2.0.
        entropy_regularization (bool): Flag indicating whether to apply entropy
            regularization to the principal objective.
        entropy_anneal (float): Same as 'anneal_speed' but for annealing the entropy
            coefficient.
        entropy_begin (int): Same as 'u_begin' but for annealing the entropy
            coefficient.
        should_plot (bool): Whether to generate plots of the training history
            (these would be updated throughout training).
    """
    
    principal_profit_multiplier = params_dict['principal_profit_multiplier']
    n_a_outputs = params_dict['n_a_outputs']
    entropy_lambda = params_dict['entropy_lambda']

    #Create a dictionary of arguments for batch_game function
    arguments_dict = {
        # 'get_principal_reward': get_principal_reward_stochastic_schedule,
        'pi_p': pi_p, 
        'optim_p': optim_p,
        'history': history,
        'mi_classifier': mi_classifier,
        'params_dict': params_dict,
        # 'characterize_func':characterize,
        'a_o_mat': a_o_mat,
        'n_batch': n_batch,
    }

    ###############
    #Lists of training visualization items to plot, 's' is a not-the-best way of indicating a smoothed version of the previous item should be plotted
    if entropy_lambda == 0: 
        plot_keys = [
            [
                ['mi_penalty', 's' ],
                ['m_entropy_mean', 'm_entropy_std'],
                ['mi_lambda']

            ],
            [
                ['r_a', 's', 'r_p', 's', 'r_p_0'],
                ['agent_action_dist'],
                ['principal_mean']
            ]
        ]
    else:
        plot_keys = [
            [
                ['e_penalty', 's' ],
                ['m_entropy_mean', 'm_entropy_std'],
                ['r_a', 's', 'r_p', 's', 'r_p_0']

            ],
            [
                
                ['agent_action_dist'],
                ['principal_mean'],
                ['principal_std']
            ]
        ]

    if should_plot:
        fig, axes = plt.subplots(len(plot_keys), len(plot_keys[0]), figsize=(16, 5*len(plot_keys)))
        r_p_0, _ = get_principal_reward_deterministic_schedule(np.zeros(n_a_outputs), a_o_mat = a_o_mat, **params_dict)
        enum_u = range(u_begin, u_begin + n_timesteps)
    else:
        enum_u = tqdm.tqdm(range(u_begin, u_begin + n_timesteps))

    for u in enum_u:
        if anneal_mi:
            mi_lambda_time = min(mi_lambda, (anneal_speed/10000.0)*u)
        else:
            mi_lambda_time = mi_lambda
        
        if entropy_regularization:
            entropy = max(0, entropy_begin - (entropy_anneal/10000.0)*u)
            pi_p.set_entropy_reg(entropy)

        #Run one round of training
        batch_game(
            updates=1, mi_lambda = mi_lambda_time,
            **arguments_dict
        )
        if should_plot:
            #Plot training visualization
            if (u+1)%200 == 0:
                for ax_sub, ks_sub in zip(axes, plot_keys):
                    for ax, ks in zip(ax_sub, ks_sub):
                        ax.cla()
                        for k in ks:
                            if k == 's':
                                ax.plot(*smooth_plot(history[last_k][100:]),
                                        label=last_k + ' smoothed');
                            elif k == 'r_p':
                                ax.plot(history[k][100:], label=k, alpha = 0.5);
                                last_k = k
                            elif k == 'r_p_0':
                                ax.hlines(r_p_0, xmin = 0, xmax = u - 100,linestyles='dashed', label = 'zero pay reward')
                            elif k in ['agent_action_dist', 'output_dist', 'principal_std']:
                                ax.imshow(np.array(history[k]).squeeze().T, aspect = 'auto', interpolation = 'none', vmin = 0)
                                ax.set_title(k)
                            elif k in [ 'principal_mean']:
                                vs = np.array(history[k]).squeeze().T
                                ax.imshow(vs, aspect='auto', interpolation='none',
                                          vmin=0,
                                          vmax=np.max(
                                              vs[:, -vs.shape[1] // 2].flatten()))
                                ax.set_title(k)
                            else:
                                ax.plot(history[k][100:], label=k);
                                last_k = k
                        if k not in ['agent_action_dist', 'output_dist', 'principal_mean', 'principal_std']:
                            ax.legend(loc='upper left');
                        ax.grid()

                display.clear_output(wait=True)
                display.display(fig)
    return None

#Function for plotting a moving window average of a noisy plot
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

#Plot the history afterwards
def plot_history(history, entropy_lambda = 0, n_a_outputs = None, a_o_mat = None, params_dict = None):
    
    if entropy_lambda == 0: 
        plot_keys = [
            [
                ['mi_penalty', 's' ],
                ['m_entropy_mean', 'm_entropy_std'],
                ['mi_lambda']

            ],
            [
                ['r_a', 's', 'r_p', 's', 'r_p_0'],
                ['agent_action_dist'],
                ['principal_mean']
            ]
        ]
    else:
        plot_keys = [
            [
                ['log_probs', 's' ],
                ['m_entropy_mean', 'm_entropy_std'],
                ['mi_lambda']

            ],
            [
                ['r_a', 's', 'r_p', 's', 'r_p_0'],
                ['agent_action_dist'],
                ['principal_mean']
            ]
        ]
        
    fig, axes = plt.subplots(len(plot_keys), len(plot_keys[0]), figsize=(16, 5*len(plot_keys)))
    if n_a_outputs is None:
        r_p_0 = None
    else:
        r_p_0, _ = get_principal_reward_deterministic_schedule(np.zeros(n_a_outputs), a_o_mat = a_o_mat, **params_dict)
    
    for ax_sub, ks_sub in zip(axes, plot_keys):
        for ax, ks in zip(ax_sub, ks_sub):
            ax.cla()
            for k in ks:
                if k == 's':
                    ax.plot(*smooth_plot(history[last_k][100:]),
                            label=last_k + ' smoothed');
                elif k == 'r_p':
                    ax.plot(history[k][100:], label=k, alpha = 0.5);
                    last_k = k
                elif k == 'r_p_0':
                    if r_p_0 is not None:
                        u = len(history[last_k][100:])
                        ax.hlines(r_p_0, xmin = 0, xmax = u - 100,linestyles='dashed', label = 'zero pay reward')
                elif k in ['agent_action_dist', 'output_dist']:
                    ax.imshow(np.array(history[k]).squeeze().T, aspect = 'auto', interpolation = 'none', vmin = 0)
                    ax.set_title(k)
                elif k in [ 'principal_mean']:
                    vs = np.array(history[k]).squeeze().T
                    ax.imshow(vs, aspect = 'auto', interpolation = 'none', vmin = 0,
                              vmax = np.max(vs[:, -vs.shape[1]//2].flatten()) )
                    ax.set_title(k)
                else:
                    ax.plot(history[k][100:], label=k)
                    last_k = k
            if k not in ['agent_action_dist', 'output_dist', 'principal_mean']:
                ax.legend(loc='upper left')
            ax.grid()

"""Saving and loading."""
#Get the prefix of save name where the save name incorporates various parameters of the experiment               
def get_name_noseed(save_folder_experiments, mi_lambda, entropy_lambda, qtemp_agent, agent_work_multiplier, principal_profit_multiplier, random_type, random_factor, agent_exp,  **kwargs):
    if entropy_lambda > 0:
        mi_name = f'e{entropy_lambda:.2f}'
    else:
        mi_name = f'mi{mi_lambda:.2f}'
    q_name = f'q{qtemp_agent:.2f}'
    agent_work_name = f'awm{agent_work_multiplier:.2f}'
    rho_name = f'rho{agent_exp:.2f}'
    principal_profit_name = f'ppm{principal_profit_multiplier:.2f}'
    random_name = f'{random_type}{random_factor:.2f}'
    truncated_name = ''
    
    save_file_name = f'{mi_name}_{q_name}_{agent_work_name}_{rho_name}_{principal_profit_name}_{random_name}{truncated_name}' #name of folder for this run
    return f'{save_folder_experiments}/{save_file_name}'

def get_processed_name(save_folder_experiments, qtemp_agent, agent_work_multiplier, principal_profit_multiplier, random_type, random_factor, agent_exp, **kwargs):
    
    q_name = f'q{qtemp_agent:.2f}'
    agent_work_name = f'awm{agent_work_multiplier:.2f}'
    rho_name = f'rho{agent_exp:.2f}'
    principal_profit_name = f'ppm{principal_profit_multiplier:.2f}'
    random_name = f'{random_type}{random_factor:.2f}'
    truncated_name = ''
    
    save_file_name = f'{q_name}_{agent_work_name}_{rho_name}_{principal_profit_name}_{random_name}{truncated_name}' #name of folder for this run
    return f'{save_folder_experiments}/{save_file_name}'

#Get name of specific seed
def get_name(save_folder_experiments, mi_lambda, random_seed, params_dict, **kwargs):
    
    seed_name = f'seed{int(random_seed)}'
    name_no_seed = get_name_noseed(save_folder_experiments, mi_lambda, **params_dict)
    return f'{name_no_seed}_{seed_name}'

#Save principal with name given the run parameters
def save_principal(pi_p, save_folder_experiments, mi_lambda, random_seed, params_dict):
    principal_model_save_name = get_name(save_folder_experiments = save_folder_experiments, mi_lambda = mi_lambda, random_seed = random_seed, params_dict = params_dict) + '_pi_p.pt'
    torch.save(pi_p.state_dict(), principal_model_save_name)
    print('Principal model saved at:\n\t{}'.format(principal_model_save_name))
    return None

#Load in a saved principal given the save name
def load_principal(principal_model_save_name, n_a_outputs, ptype = 'G'):
    if ptype == 'G':
        pi_p = PGBanditMVGModel(n_a_outputs = n_a_outputs)
    else:
        pi_p = PGBanditLNModel(n_a_outputs = n_a_outputs)
    pi_p.load_state_dict(torch.load(principal_model_save_name))
    pi_p.eval()
    
    return pi_p

#Save miclassifier with name given the run parameters
def save_mi_classifier(mi_classifier, save_folder_experiments, mi_lambda, random_seed, params_dict):
    if mi_classifier is not None:
        mi_classifier_model_save_name = get_name(save_folder_experiments = save_folder_experiments, mi_lambda = mi_lambda, random_seed = random_seed,  params_dict = params_dict) + '_mi_classifier.pt'
        torch.save(mi_classifier.state_dict(), mi_classifier_model_save_name)
        print(
            'MI classifier model saved at:\n\t{}'.format(mi_classifier_model_save_name)
        )
    else:
        print("No MI classifier to save")
    
    return None

#Load in the mi classifier
def load_mi_classifier(mi_classifier_save_name):
    mi_classifier = MIClassifier()
    mi_classifier.load_state_dict(torch.load(mi_classifier_save_name))
    mi_classifier.eval()
    return mi_classifier

def save_principal_optimizer(optim_p, save_folder_experiments, mi_lambda, random_seed, params_dict):
    optim_principal_model_save_name = get_name(save_folder_experiments = save_folder_experiments, mi_lambda = mi_lambda, random_seed = random_seed, params_dict = params_dict) + '_optim_p.pt'
    torch.save(optim_p.state_dict(), optim_principal_model_save_name)
    print(
        'Principal model optimizer saved at:\n\t{}'.format(
            optim_principal_model_save_name
        )
    )
    return None

def save_partial_history(history_dict, save_folder_experiments, mi_lambda, random_seed, params_dict, num_save = -1):
    history_model_save_name = get_name(save_folder_experiments = save_folder_experiments, mi_lambda = mi_lambda, random_seed = random_seed,  params_dict = params_dict) + '_hist.pickle'

    if num_save < 0:
        history_save = history_dict
    else:
        history_save = {key:value[-num_save:] for key, value in history_dict.items()}
        
    with open(history_model_save_name, 'wb') as f:
        pickle.dump(history_save, f)

    print('Training history saved at:\n\t{}'.format(history_model_save_name))
    
    return None

