import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class MutualInformationClassifier(nn.Module):
    """
    Classifier used to estimate I(X,Y).

    Classification is trained to discriminate samples from p(x, y) versus p(x)p(y).
    The `log_odds_ratio` method can be used for estimating MI cost.
    The `forward` method provides a training loss.

    Args:
        x_dim (int): Dimension size of the x input.
        y_dim (int): Dimension size of the y input.
        h_dims (int, float, tuple, list): Defaults to (64,). Sets the hidden size(s)
            of the encoder layer(s). If an int or float, interpreted as the hidden
            size of a single hidden layer net.
    """

    def __init__(self, x_dim, y_dim, h_dims=(64,)):
        super().__init__()

        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)

        if isinstance(h_dims, (int, float)):
            self.h_dims = [int(h_dims)]
        elif isinstance(h_dims, (list, tuple)):
            self.h_dims = list(h_dims)
        else:
            raise TypeError

        layer_list = []

        last_h_dim = self.x_dim + self.y_dim
        for h_dim in self.h_dims:
            layer_list.append(nn.Linear(last_h_dim, h_dim))
            layer_list.append(nn.ReLU())
            last_h_dim = int(h_dim)

        layer_list.append(nn.Linear(last_h_dim, 2))

        self.logits = nn.Sequential(*layer_list)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def log_odds_ratio(self, xs, ys):
        """This is the basis of the MI cost!"""
        xy = torch.cat([xs.view(-1, self.x_dim), ys.view(-1, self.y_dim)], 1)
        log_probs = torch.log(self.softmax(self.logits(xy)))
        # Clipping prevents nan-causing values at OOD x,y pairs.
        clipped_log_odds_ratio = torch.clamp(
            log_probs[:, 1] - log_probs[:, 0],
            min=-50.0,
            max=50.0
        ).detach().numpy()
        return clipped_log_odds_ratio

    def predict(self, xs, ys):
        """Gives probability that x,y pairs come from p(x,y) versus p(x)p(y)"""
        xy = torch.cat([xs.view(-1, self.x_dim), ys.view(-1, self.y_dim)], 1)
        probs = self.softmax(self.logits(xy))
        return probs[:, 1]

    def forward(self, xs, ys):
        """Given a batch of xs and corresponding ys, returns training loss."""
        xs = xs.view(-1, self.x_dim).detach()
        ys = ys.view(-1, self.y_dim).detach()

        xy_from_policy = torch.cat([xs, ys], dim=1)
        xy_from_marginals = torch.cat([xs, ys[torch.randperm(ys.shape[0])]], dim=1)
        xy = torch.cat([xy_from_policy, xy_from_marginals], dim=0)

        targets = torch.cat(
            [torch.ones(ys.shape[0], dtype=torch.long),
             torch.zeros(ys.shape[0], dtype=torch.long)],
            dim=0
        )

        return self.loss(self.logits(xy), targets).mean()


class PerceptualEncoder(nn.Module):
    """
    Encodes input X into stochastic Y.

    Learns p(y | x, aux), where aux is an optional auxiliary input.
    The auxiliary input is separate from x in case we care specifically about I(X,Y).

    The `encode` method samples from p(y | x, aux) and gives the log_prob of the sample.
    The `forward` method provides a policy-gradient training loss.

    Args:
        x_dim (int): The dimension size of the x input.
        aux_dim (optional, int): If supplied, the dimension size of the auxiliary input.
            The encoding "y" will be a function of the concatenation of "x" and "aux"
            inputs, if aux is not None.
        residual_style (bool): Defaults to True. Whether to add the stochastic
            encoding to the "x" input. If chosen, the encoding dimension ("y_dim") is
            automatically set to match "x_dim".
        y_dim (optional, int): Only used if residual_style=False, in which case you
            must supply this argument, as it determines the dimension size of the
            output encoding "y".
        h_dims (int, float, list, tuple): Defaults to (64,). Sets the hidden size(s)
            of the encoder layer(s). If an int or float, interpreted as the hidden
            size of a single hidden layer net.
        log_std_bias (float): Defaults to -3.0. A bias term to manually add to the
            STD of the output, before applying the softplus nonlinearity. A negative
            value helps to ensure that sampled encodings "y" are not overly noisy
            during the early phases of training. This is empirically helpful when "y" is
            used as an input to a decision policy.
    """

    def __init__(self,
                 x_dim,
                 aux_dim=None,
                 residual_style=True,
                 y_dim=None,
                 h_dims=(64,),
                 log_std_bias=-3.0
                 ):
        super().__init__()

        self.x_dim = int(x_dim)
        assert self.x_dim > 0

        if isinstance(aux_dim, (int, float)):
            self.aux_dim = int(aux_dim)
        elif aux_dim is None:
            self.aux_dim = 0
        else:
            raise TypeError
        assert self.aux_dim >= 0

        self.residual_style = bool(residual_style)

        if self.residual_style:
            self.y_dim = self.x_dim
        else:
            self.y_dim = int(y_dim)
            assert self.y_dim > 0

        if isinstance(h_dims, (int, float)):
            self.h_dims = [int(h_dims)]
        elif isinstance(h_dims, (list, tuple)):
            self.h_dims = list(h_dims)
        else:
            raise TypeError

        self.log_std_bias = float(log_std_bias)
        assert -5.0 <= self.log_std_bias <= 5.0

        layer_list = []

        last_h_dim = self.x_dim + self.aux_dim
        for h_dim in self.h_dims:
            layer_list.append(nn.Linear(last_h_dim, h_dim))
            layer_list.append(nn.ReLU())
            last_h_dim = int(h_dim)

        layer_list.append(
            nn.Linear(last_h_dim, self.y_dim * 2)
            # Output mean and pre-softplus-STD for each y (output) dimension
        )

        self.encoded_params = nn.Sequential(*layer_list)
        self.softplus = nn.Softplus()

    def get_encoding_mean_and_std(self, xs, auxs=None):
        xs = xs.view(-1, self.x_dim)

        if auxs is None:
            inp = xs
        else:
            assert self.aux_dim > 0
            aux_inp = auxs.view(-1, self.aux_dim)
            inp = torch.cat([xs, aux_inp], 1)

        mean_and_log_std = self.encoded_params(inp)

        y_mean = mean_and_log_std[:, :self.y_dim]
        y_std = self.softplus(mean_and_log_std[:, self.y_dim:] + self.log_std_bias)

        if self.residual_style:
            y_mean = y_mean + xs
        y_std = torch.clamp(y_std, 0.05, 100)
        y_mean = torch.clamp(y_mean, 0.0, 1000) #TODO: Remove this, I think the clamp of y_std to greater than 0.05 is all that is necessary
        return y_mean, y_std

    def encode(self, xs, auxs=None):
        y_mean, y_std = self.get_encoding_mean_and_std(xs=xs, auxs=auxs)
        try:
            m = Normal(y_mean, y_std)
        except:
            print(y_mean)
            print(y_std)

        # IMPORTANT!!! Gradients can flow through the sampled ys (because `rsample`)
        ys = m.rsample()

        # Necessary to detach ys here to get proper gradient flow.
        enc_log_probs = m.log_prob(ys.detach()).sum(dim=1)

        return ys, enc_log_probs

    def forward(self, enc_log_probs, advantages):
        losses = -enc_log_probs * advantages.detach()
        return losses.mean()


class CostlyPerceptualEncoder(nn.Module):
    """
    Combines PerceptualEncoder instance with dedicated MutualInformationClassifier
    instance to (1) generate perceptual encodings of inputs and (2) maintain an
    estimate of the MI of the encodings, all of which is trainable.

    Args:
        x_dim (int): See "x_dim" of PerceptionEncoder
        mi_h_dims (int, float, list, tuple): Passed as the "h_dims" argument to the
            MutualInformationClassifier.
        mi_cost (float): Defaults to 0.0. The scale applied to the MI estimate when
            outputing MI cost of the sampled encoding "y" given "x".
        aux_to_mi (bool): Defaults to True. If True, the auxiliary inputs (if any)
            are treated as part of the "x" input to the MI classifier.
            This changes the interpretation of MI cost!!!
        **perceptual_encoder_kwargs: Keyword arguments of PerceptualEncoder that will be
            passed to it when constructing the PerceptualEncoder instance.
    """

    def __init__(self, x_dim, mi_h_dims=(64,), mi_cost=0.0, aux_to_mi=True,
                 **perceptual_encoder_kwargs):
        super().__init__()

        self.x_dim = int(x_dim)
        assert self.x_dim > 0

        self.mi_cost = float(mi_cost)
        assert self.mi_cost >= 0.0

        self.aux_to_mi = bool(aux_to_mi)

        self.perceptual_encoder = PerceptualEncoder(
            self.x_dim,
            **perceptual_encoder_kwargs
        )

        self.aux_dim = self.perceptual_encoder.aux_dim

        self.y_dim = int(self.perceptual_encoder.y_dim)
        assert self.y_dim > 0

        self.mi_classifier = MutualInformationClassifier(
            self.x_dim + (self.aux_dim if self.aux_to_mi else 0),
            self.y_dim,
            mi_h_dims,
        )

    def get_encoder_params(self):
        return self.perceptual_encoder.parameters()

    def get_mi_classifier_params(self):
        return self.mi_classifier.parameters()

    def get_encoding_mean_and_std(self, xs, auxs=None):
        return self.perceptual_encoder.get_encoding_mean_and_std(xs=xs, auxs=auxs)

    def encode(self, xs, auxs=None):
        # Sample a (stochastic) encoding of the input x (can be conditioned on
        # auxiliary input)
        ys, enc_log_probs = self.perceptual_encoder.encode(xs, auxs)

        # Estimate the MI cost associated with each encoding.
        if self.aux_to_mi and self.aux_dim > 0:
            assert auxs is not None
            mi_xs = torch.cat([xs.view(-1, self.x_dim), auxs.view(-1, self.aux_dim)], 1)
        else:
            mi_xs = xs
        mi_cost = self.mi_classifier.log_odds_ratio(mi_xs, ys) * self.mi_cost

        return ys, enc_log_probs, mi_cost

    def forward(self, mi_xs, ys, log_probs, advantages):
        # Note, assumption here is that auxiliary inputs have been concatenated with
        # xs, if appropriate
        mi_classifier_loss = self.mi_classifier(mi_xs, ys)

        encoder_loss = self.perceptual_encoder(log_probs, advantages)

        return mi_classifier_loss + encoder_loss


class MultiChannelCostlyPerceptualEncoder(nn.Module):
    """
    Generates a single stochastic percept from multiple perceptual channels.

    Note: If an auxiliary input is used, it will be used by each perceptual encoder.

    Args:
        aux_dim (optional, int): Dimension size of any auxiliary input that will be
            fed as an input to each channel-specific costly encoder.
        **channel_dicts (dictionary): Each key is the name of a channel, and each
            value is the kwargs dictionary for the CostlyPerceptualEncoder instance
            associated with that channel.
            NOTE: The names of the channels will need to match the names of the xs,
            as shown in the example.

    Example:
        # Make an instance with 2 channels, no auxilairy input
        mccpe = MultiChannelCostlyPerceptualEncoder(
            channel_dicts={
                'a': dict(x_dim=5, log_std_bias=0.0, mi_cost=1.0),
                'b': dict(x_dim=3, residual_style=False, y_dim=10, mi_cost=2.0)
            }
        )

        # Encode some channel-separated input
        obs_a, obs_b = ...
        ys, mi_costs, training_dict, avg_mi_cost_channel = mccpe.encode({'a': obs_a, 'b': obs_b})
    """

    def __init__(self, aux_dim=None, channel_dicts={}):
        super().__init__()

        if isinstance(aux_dim, (int, float)):
            self.aux_dim = int(aux_dim)
        elif aux_dim is None:
            self.aux_dim = 0
        else:
            raise TypeError
        assert self.aux_dim >= 0

        assert isinstance(channel_dicts, dict)
        self.encoders = nn.ModuleDict()
        for channel_name, channel_kwargs in channel_dicts.items():
            if 'aux_dim' in channel_kwargs:
                if self.aux_dim == 0:
                    assert channel_kwargs['aux_dim'] in [0, None]
                else:
                    assert channel_kwargs['aux_dim'] == self.aux_dim
            channel_kwargs['aux_dim'] = self.aux_dim

            self.encoders[channel_name] = CostlyPerceptualEncoder(**channel_kwargs)

        assert len(self.encoders) >= 1

        self._channels = [k for k in self.encoders.keys()]

        self.y_dim = sum(encoder.y_dim for encoder in self.encoders.values())

    @property
    def channels(self):
        return self._channels

    def get_encoder_params(self):
        p = []
        for k in self.channels:
            p += self.encoders[k].get_encoder_params()
        return p

    def get_mi_classifier_params(self):
        p = []
        for k in self.channels:
            p += self.encoders[k].get_mi_classifier_params()
        return p

    def encode(self, xs_dict, auxs=None):
        ys = []
        mi_cost_list = []
        avg_mi_cost_channel = []
        training_dict = {}

        for channel in self.channels:
            encoder = self.encoders[channel]
            encoder_xs = xs_dict[channel]
            encoder_ys, encoder_log_probs, encoder_mi_cost = encoder.encode(
                encoder_xs, auxs
            )
            avg_mi_cost_channel.append((channel, np.mean(encoder_mi_cost)))

            ys.append(encoder_ys)
            mi_cost_list.append(encoder_mi_cost)

            if encoder.aux_to_mi:
                if self.aux_dim == 0:
                    encoder_mi_xs = encoder_xs
                else:
                    assert auxs is not None
                    encoder_mi_xs = torch.cat([encoder_xs.view(-1, encoder.x_dim),
                                               auxs.view(-1, self.aux_dim)], 1)
            else:
                encoder_mi_xs = encoder_xs

            training_dict[channel] = {
                'mi_xs': encoder_mi_xs,
                'ys': encoder_ys,
                'log_probs': encoder_log_probs
            }

        # Concatenate the channel-wise encodings
        ys = torch.cat(ys, dim=1)

        # Add together the channel-wise MI costs
        mi_costs = np.stack(mi_cost_list).sum(axis=0)

        return ys, mi_costs, training_dict, avg_mi_cost_channel

    def forward(self, advantages, training_dict):
        loss = 0
        for channel in self.channels:
            loss += self.encoders[channel](
                advantages=advantages, **training_dict[channel]
            )
        return loss
                                       


class MultiChannelCostlyPerceptualEncoderDiscretePolicy(nn.Module):
    """
    Adds a discrete action head on top of MultiChannelCostlyPerceptualEncoder

    Args:
        action_dim (int): The number of actions that can be taken.
        n_actions (int): Defaults to 1, The number of actions to output.
        action_mi_cost (float): Defaults to 0.0. The scale applied to the MI estimate
            when outputing MI cost of the sampled action "a" given "y".
        entropy_coeff (float): Defaults to 0.0. Coefficient placed on the action
            entropy for regularizing the policy. You may get weird effects if using
            BOTH an MI and entropy cost on the policy.
        h_dims (int, float, list, tuple): Defaults to (64, 64). Sets the hidden size(s)
            of the action head layer(s). If an int or float, interpreted as the hidden
            size of a single hidden layer net.
        channel_dicts (dictionary): see MultiChannelCostlyPerceptualEncoder

    Example:
        # Make an instance with 2 channels
        mccpe_pol = MultiChannelCostlyPerceptualEncoderDiscretePolicy(
            action_dim=2,
            action_mi_cost=0.1,
            **{
                'a': dict(x_dim=5, log_std_bias=0.0, mi_cost=1.0),
                'b': dict(x_dim=3, residual_style=False, y_dim=10, mi_cost=2.0)
            }
        )

        # Sample actions, and total MI costs of internal encodings and the actions
        obs_a, obs_b = ...
        actions, mi_costs, a_log_probs, a_ents, training_dict = mccpe_pol.sample(
            {'a': obs_a, 'b': obs_b}
        )
        """
    def __init__(self,
                 action_dim,
                 n_actions = 1,
                 action_mi_cost=0.0,
                 entropy_coeff=0.0,
                 h_dims=(64, 64),
                 channel_dicts={}
                 ):
        super().__init__()
        self.n_actions = int(n_actions)
        assert self.n_actions >= 1
        
        
        self.action_dim = int(action_dim)
        assert self.action_dim >= 2
        if self.n_actions > 1:
            self.action_dim = self.action_dim * self.n_actions
        
        
        

        self.action_mi_cost = float(action_mi_cost)
        assert self.action_mi_cost >= 0.0

        self.entropy_coeff = float(entropy_coeff)
        assert self.entropy_coeff >= 0.0

        if isinstance(h_dims, (int, float)):
            self.h_dims = [int(h_dims)]
        elif isinstance(h_dims, (list, tuple)):
            self.h_dims = list(h_dims)
        else:
            raise TypeError

        self.mccpe = MultiChannelCostlyPerceptualEncoder(
            aux_dim=self.aux_dim,
            channel_dicts=channel_dicts
        )

        layer_list = []

        last_h_dim = self.action_head_input_dim
        for h_dim in self.h_dims:
            layer_list.append(nn.Linear(last_h_dim, h_dim))
            layer_list.append(nn.ReLU())
            last_h_dim = int(h_dim)

        layer_list.append(
            nn.Linear(last_h_dim, self.action_dim)
            # Output logits based on LSTM state + last encoded outputs
        )

        self.logits = nn.Sequential(*layer_list)

        self.action_mi_classifier = MutualInformationClassifier(
            x_dim=self.action_head_input_dim, y_dim=self.n_actions
        )

        self._optimizer = None

    @property
    def aux_dim(self):
        return 0

    @property
    def action_head_input_dim(self):
        return self.mccpe.y_dim + self.aux_dim

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam([
                {'params': self.get_policy_params(), 'lr': 0.0001},
                {'params': self.get_mi_classifier_params(), 'lr': 0.001}
            ])
        return self._optimizer

    @property
    def channels(self):
        return self.mccpe.channels

    def get_policy_params(self):
        p = []
        p += self.mccpe.get_encoder_params()
        p += self.logits.parameters()
        return p

    def get_mi_classifier_params(self):
        p = []
        p += self.mccpe.get_mi_classifier_params()
        p += self.action_mi_classifier.parameters()
        return p

    def sample(self, xs_dict):
        # Sample encodings of the inputs, encoding MI costs, and data for training
        ys, encoding_mi_costs, training_dict, avg_mi_cost_channel = self.mccpe.encode(xs_dict)

        # Run ys through logits net to get action dist, then sample
        
        logits = self.logits(ys)
        
        m = Categorical(logits=logits)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        action_entropies = m.entropy()

        # Add in the MI cost for the actions.
        action_mis = self.action_mi_classifier.log_odds_ratio(ys, actions)
        action_mi_costs = self.action_mi_cost * action_mis
        total_mi_costs = encoding_mi_costs + action_mi_costs

        # Incorporate action head training stuffs
        training_dict['_action_head'] = {
            'mi_xs': ys,
            'ys': actions,
            'log_probs': action_log_probs,
            'entropies': action_entropies,
        }

        actions = actions.detach().numpy()

        return actions, total_mi_costs, training_dict

    def forward(self, advantages, training_dict):
        pg_losses = -training_dict['_action_head']['log_probs'] * advantages
        ent_losses = -training_dict['_action_head']['entropies'] * self.entropy_coeff
        policy_loss = pg_losses.mean() + ent_losses.mean()

        action_mi_classifier_loss = self.action_mi_classifier(
            training_dict['_action_head']['mi_xs'],
            training_dict['_action_head']['ys'],
        )

        mcppe_loss = self.mccpe(advantages, training_dict)

        return policy_loss + action_mi_classifier_loss + mcppe_loss
                                       


class MultiChannelCostlyPerceptualEncoderRecurrentDiscretePolicy(
    MultiChannelCostlyPerceptualEncoderDiscretePolicy
):
    """
    Extends MultiChannelCostlyPerceptualEncoderDiscretePolicy to use an LSTM to
    maintain a learnable memory of encodings. LSTM state becomes an auxiliary input
    to encoders and an additional input the the action head.

    Args:
        *args: See MultiChannelCostlyPerceptualEncoderDiscretePolicy
        lstm_dim (int): Dimension size of LSTM hidden state
        **kwargs: See MultiChannelCostlyPerceptualEncoderDiscretePolicy
    """
    def __init__(self, *args, lstm_dim=32, **kwargs):

        self.lstm_dim = int(lstm_dim)
        assert self.lstm_dim > 0

        super().__init__(*args, **kwargs)

        self.lstm_initial_state_h = nn.Parameter(torch.zeros(1, self.lstm_dim))
        self.lstm_initial_state_c = nn.Parameter(torch.zeros(1, self.lstm_dim))
        self.lstm_cell = nn.LSTMCell(
            input_size=self.mccpe.y_dim,
            hidden_size=self.lstm_dim
        )

    @property
    def aux_dim(self):
        return self.lstm_dim

    def get_policy_params(self):
        p = []
        p += self.mccpe.get_encoder_params()
        p += self.logits.parameters()
        p += self.lstm_cell.parameters()
        p += [self.lstm_initial_state_h, self.lstm_initial_state_c]
        return p

    def get_initial_state_tuple(self, batch_size):
        lstm_state_h = torch.tile(self.lstm_initial_state_h, (batch_size, 1))
        lstm_state_c = torch.tile(self.lstm_initial_state_c, (batch_size, 1))
        initial_lstm_state = (lstm_state_h, lstm_state_c)
        return initial_lstm_state

    def sample(self, xs_dict, lstm_state=None):
        # Use the (learnable) initial state if none are provided
            
        if lstm_state is None:
            batch_size = list(xs_dict.values())[0].shape[0]
            lstm_state = self.get_initial_state_tuple(batch_size)

        # Sample encodings of the inputs, MI costs of the encodings, and training data
        ys, encoding_mi_costs, training_dict, avg_mi_cost_channel = self.mccpe.encode(
            xs_dict, auxs=lstm_state[0]
        )

        # Run ys through LSTM to get updated states
        lstm_state = self.lstm_cell(ys, lstm_state)
        # Run ys + updated states through logits net to get action dist, then sample
        action_head_inputs = torch.cat([ys, lstm_state[0]], 1)
        logits = self.logits(action_head_inputs)
        if self.n_actions == 1:
            m = Categorical(logits=logits)
            actions = m.sample()
            action_log_probs = m.log_prob(actions)
            action_entropies = m.entropy()
        else:
            n_batch, n_actions_dim = logits.shape
            logits_r = logits.view(n_batch, self.n_actions, -1)
            m = Categorical(logits = logits_r)
            actions = m.sample()
            
            action_log_probs_separate = m.log_prob(actions)
            action_log_probs = action_log_probs_separate.sum(1)
            action_entropies_separate = m.entropy()
            action_entropies = action_entropies_separate.sum(1)
            

        # Add in the MI cost for the actions.
        action_mis = self.action_mi_classifier.log_odds_ratio(
            action_head_inputs, actions
        )
        action_mi_costs = self.action_mi_cost * action_mis
        total_mi_costs = encoding_mi_costs + action_mi_costs
        
        avg_mi_cost_channel.append(('action', np.mean(action_mi_costs)))

        # Incorporate action head training stuffs
        training_dict['_action_head'] = {
            'mi_xs': action_head_inputs,
            'ys': actions,
            'log_probs': action_log_probs,
            'entropies': action_entropies,
        }

        actions = actions.detach().numpy()

        return actions, total_mi_costs, training_dict, lstm_state, m, avg_mi_cost_channel
                                       
    
#     def get_logits(self, xs_dict, lstm_state = None):
#         if lstm_state is None:
#             batch_size = list(xs_dict.values())[0].shape[0]
#             lstm_state = self.get_initial_state_tuple(batch_size)

#         # Sample encodings of the inputs, MI costs of the encodings, and training data
#         ys, encoding_mi_costs, training_dict = self.mccpe.encode(
#             xs_dict, auxs=lstm_state[0]
#         )

#         # Run ys through LSTM to get updated states
#         lstm_state = self.lstm_cell(ys, lstm_state)

#         # Run ys + updated states through logits net to get action dist, then sample
#         action_head_inputs = torch.cat([ys, lstm_state[0]], 1)
#         logits = self.logits(action_head_inputs)
#         return logits


#Wrapper Agent for taking care of some training/saving and loading details for the MultiChannelCostlyPerceptualEncoderRecurrentDiscretePolicy
class MCCPERDPAgent():
    def __init__(self, action_dim,
                 n_actions = 1,
                 action_mi_cost=0.0,
                 entropy_coeff=0.0,
                 h_dims=(64, 64),
                 channel_dicts={},
                 lstm_dim=32, 
                 future_rewards_only = True,
                 clip_grads = False
                ):

        self.mccperdp = MultiChannelCostlyPerceptualEncoderRecurrentDiscretePolicy(
            action_dim=action_dim,
            n_actions = n_actions,
            h_dims=h_dims,
            entropy_coeff=entropy_coeff,
            lstm_dim=lstm_dim,
            action_mi_cost=action_mi_cost,
            channel_dicts=channel_dicts,

        )
        self.lstm_state = None
        self.batch_size = 0
        self.future_rewards_only = future_rewards_only
        self.clip_grads = clip_grads

        self._train_counter = 0

    @property
    def training_iters(self):
        return self._train_counter
    
    #clears the training dict and lstm state for a new episode
    def new_episode(self):
        self.lstm_state = None
        self.training_dict = {}
        self.mi_channel_through_time = {}
                                       
        
    def act(self, p_state, get_model = False):
        p_actions, total_mi_costs, training_dict_, lstm_state, m, avg_mi_cost_channel = self.mccperdp.sample(
            p_state,
            lstm_state=self.lstm_state,
        )
        
        for channel_name, mi_cost in avg_mi_cost_channel:
            if channel_name not in self.mi_channel_through_time:
                self.mi_channel_through_time[channel_name] = []
            self.mi_channel_through_time[channel_name].append(mi_cost)
        

        for k, v in training_dict_.items():
            if k not in self.training_dict:
                self.training_dict[k] = {vk: [] for vk in v.keys()}
            for vk, vv in v.items():
                self.training_dict[k][vk].append(vv)
        self.lstm_state = lstm_state
        self.batch_size = list(p_state.values())[0].shape[0]
        if not get_model:
            return p_actions, total_mi_costs
        else:
            return p_actions, total_mi_costs, m

    def end_episode(self, rewards_seq, train = True):
        loss = 0
        # ... aggregate tensors for the PRINCIPAL
        for channel in self.training_dict.keys():
            for k, v in self.training_dict[channel].items():
                if k in ['mi_xs', 'ys']:
                    self.training_dict[channel][k] = torch.cat(v, 0)
                else:
                    self.training_dict[channel][k] = torch.stack(v)

        if train:
            # ... calculate cumulative rewards & advantages for the PRINCIPAL
            rs = np.stack(rewards_seq)
            if self.future_rewards_only:
                cum_rs = np.cumsum(rs[::-1], axis=0)[::-1]  # cumulative rewards are [timesteps, batch_size]
            else:
                n_timesteps = len(rewards_seq)
                cum_rs = np.tile(np.sum(rs, axis = 0),(n_timesteps, 1)) #use sum of rewards for all timesteps
            advantages = torch.as_tensor(
                (cum_rs - cum_rs.mean(1, keepdims=True)) / cum_rs.std(1, keepdims=True),
                dtype=torch.float32)



            #### TRAIN PRINCIPAL ####
            self.mccperdp.optimizer.zero_grad()
            loss = self.mccperdp(advantages, self.training_dict)
            loss.backward()
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.mccperdp.parameters(), 4.0)
            self.mccperdp.optimizer.step()
            self._train_counter += 1

        return loss

    def get_mis_channels(self):
        channel_mis = []

        for channel in self.mccperdp.mccpe.channels:
            mi_classifier = self.mccperdp.mccpe.encoders[channel].mi_classifier
            log_odds = mi_classifier.log_odds_ratio(
                self.training_dict[channel]['mi_xs'], self.training_dict[channel]['ys']
            ).reshape(-1, self.batch_size)
            channel_mis.append(('mi-' + channel, log_odds.mean(1)))
        return channel_mis
    
    def get_mis_channel_history(self):
        return self.mi_channel_through_time
    
    def save_model(self, file_prefix):
        file_name = file_prefix
        
        torch.save(self.mccperdp.state_dict(), file_name)
    
    def load_model(self, file_prefix):
        file_name = file_prefix
        self.mccperdp.load_state_dict(torch.load(file_name))