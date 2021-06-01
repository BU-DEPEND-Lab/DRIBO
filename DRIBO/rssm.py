import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np

from DRIBO.utils import namedarraytuple


RSSMState = namedarraytuple('RSSMState', ['mean', 'std', 'stoch', 'deter'])
# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


def stack_states(rssm_states: list, dim):
    return RSSMState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )


def flatten_states(rssm_states: RSSMState, batch_shape):
    return RSSMState(
        torch.reshape(rssm_states.mean, (batch_shape, -1)),
        torch.reshape(rssm_states.std, (batch_shape, -1)),
        torch.reshape(rssm_states.stoch, (batch_shape, -1)),
        torch.reshape(rssm_states.deter, (batch_shape, -1)),
    )


def get_feat(rssm_state: RSSMState):
    return torch.cat([rssm_state.stoch, rssm_state.deter], dim=-1)


def get_dist(rssm_state: RSSMState):
    return td.independent.Independent(
        td.Normal(rssm_state.mean, rssm_state.std), 1
    )


class TransitionBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev_action, prev_state):
        """
        p(s_t | s_{t-1}, a_{t-1})
        """
        raise NotImplementedError


class RepresentaionBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obs_embed, prev_action, prev_state):
        """
        p(s_t | s_{t-1}, a_{t-1}, o_t)
        """
        raise NotImplementedError


class RollOutModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, steps, obs_embed, prev_action, prev_state):
        raise NotImplementedError


class RSSMTransition(TransitionBase):
    """
    Recurrent state space model
    """
    def __init__(
        self, action_size, stochastic_size=30, deterministic_size=200,
        hidden_size=200, activation=nn.ELU, distribution=td.Normal
    ):
        super().__init__()
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._cell = nn.GRUCell(hidden_size, deterministic_size)
        self._rnn_input_model = self._build_rnn_input_model()
        self._stochastic_prior_model = self._build_stochastic_model()
        self._dist = distribution
        self._ln_stoch = nn.LayerNorm(stochastic_size)
        self._ln_deter = nn.LayerNorm(deterministic_size)

    def _build_rnn_input_model(self):
        rnn_input_model = [nn.Linear(
            self._action_size + self._stoch_size, self._hidden_size
        )]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._deter_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.LayerNorm(self._hidden_size)]
        stochastic_model += [nn.Linear(
            self._hidden_size, 2 * self._stoch_size
        )]
        return nn.Sequential(*stochastic_model)

    def initial_state(self, batch_size, **kwargs):
        return RSSMState(
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._deter_size, **kwargs),
        )

    def forward(self, prev_action: torch.Tensor, prev_state: RSSMState):
        rnn_input = self._rnn_input_model(
            torch.cat([prev_action, prev_state.stoch], dim=-1)
        )
        deter_state = self._cell(rnn_input, prev_state.deter)
        deter_state = self._ln_deter(deter_state)
        mean, std = torch.chunk(
            self._stochastic_prior_model(deter_state), 2, dim=-1
        )
        std = F.softplus(std) + 0.1
        dist = self._dist(mean, std)
        stoch_state = dist.rsample()
        stoch_state = self._ln_stoch(stoch_state)
        return RSSMState(mean, std, stoch_state, deter_state)


class RSSMRepresentation(RepresentaionBase):
    """
    Recurrent State Space Model for encoding the observation
    """
    def __init__(
        self, transition_model: RSSMTransition, obs_embed_size, action_size,
        stochastic_size=30, deterministic_size=200, hidden_size=200,
        activation=nn.ELU, distribution=td.Normal
    ):
        super().__init__()
        self._transition_model = transition_model
        self._obs_embed_size = obs_embed_size
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._dist = distribution
        self._stochastic_posterior_model = self._build_stochastic_model()
        self._ln_stoch = nn.LayerNorm(stochastic_size)

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(
            self._deter_size + self._obs_embed_size, self._hidden_size
        )]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.LayerNorm(self._hidden_size)]
        stochastic_model += [nn.Linear(
            self._hidden_size, 2 * self._stoch_size
        )]
        return nn.Sequential(*stochastic_model)

    def initial_state(self, batch_size, **kwargs):
        return RSSMState(
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._deter_size, **kwargs),
        )

    def forward(
        self, obs_embed: torch.Tensor,
        prev_action: torch.Tensor, prev_state: RSSMState
    ):
        prior_state = self._transition_model(prev_action, prev_state)
        x = torch.cat([prior_state.deter, obs_embed], dim=-1)
        mean, std = torch.chunk(self._stochastic_posterior_model(x), 2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = self._dist(mean, std)
        stoch_state = dist.rsample()
        stoch_state = self._ln_stoch(stoch_state)
        posterior_state = RSSMState(mean, std, stoch_state, prior_state.deter)
        return prior_state, posterior_state


class RSSMRollout(RollOutModule):
    """
    Collect the predictive states from RSSM
    """
    def __init__(
        self, representation_model: RSSMRepresentation,
        transition_model: RSSMTransition
    ):
        super().__init__()
        self.representation_model = representation_model
        self.transition_model = transition_model

    def forward(
        self, steps: int, obs_embed: torch.Tensor,
        action: torch.Tensor, prev_state: RSSMState
    ):
        return self.rollout_representation(
            steps, obs_embed, action, prev_state
        )

    def rollout_representation(
        self, steps: int, obs_embed: torch.Tensor,
        action: torch.Tensor, prev_state: RSSMState
    ):
        """
        Roll out the model with actions and observations from data.
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, action_size)
        :param action: size(time_steps, batch_size, action_size)
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: prior, posterior states.
        size(time_steps, batch_size, state_size)
        """
        priors = []
        posteriors = []
        for t in range(steps):
            prior_state, posterior_state = self.representation_model(
                obs_embed[t], action[t], prev_state
            )
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state
        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior, post


class ObservationEncoder(nn.Module):
    def __init__(
        self, depth=32, stride=1, shape=(3, 64, 64), output_logits=False,
        num_layers=2, feature_dim=50, activation=nn.ReLU
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(shape[0], depth, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(depth, depth, 3, stride=stride))

        out_dim = OUT_DIM_64[num_layers] \
            if shape[-1] == 64 else OUT_DIM[num_layers]
        self.embed_size = depth * out_dim * out_dim

        self.fc = nn.Linear(self.embed_size, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.shape = shape
        self.stride = stride
        self.depth = depth
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.output_logits = output_logits

    def forward_conv(self, obs):
        obs = obs / 255.

        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        return conv

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        obs = obs.reshape(-1, *img_shape)
        embed = self.forward_conv(obs)
        embed = torch.reshape(embed, (np.prod(batch_shape), -1))
        embed = self.fc(embed)
        embed = self.ln(embed)
        if not self.output_logits:
            embed = torch.tanh(embed)
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed

    def spatial_attention(self, obs):
        spatial_softmax = nn.Softmax(1)
        img_shape = obs.shape[-3:]
        gs = [None] * len(self.convs)
        x = obs
        for idx, layer in enumerate(self.convs):
            x = layer(
                x.reshape(-1, *img_shape) / 255.
            ) if idx == 0 else layer(x)
            gs[idx] = x
        gs = [gs_.pow(2).mean(1) for gs_ in gs]
        return [spatial_softmax(
            gs_.view(*gs_.size()[:1], -1)
        ).view_as(gs_) for gs_ in gs]


# class ObservationEncoder(nn.Module):
#     def __init__(
#         self, depth=32, stride=2, shape=(3, 64, 64), activation=nn.ReLU,
#         num_layers=None, feature_dim=None, output_logits=None
#     ):
#         super().__init__()
#         self.convolutions = nn.Sequential(
#             nn.Conv2d(shape[0], 1 * depth, 4, stride),
#             activation(),
#             nn.Conv2d(1 * depth, 2 * depth, 4, stride),
#             activation(),
#             nn.Conv2d(2 * depth, 4 * depth, 4, stride),
#             activation(),
#             nn.Conv2d(4 * depth, 8 * depth, 4, stride),
#             activation(),
#         )
#         self.shape = shape
#         self.stride = stride
#         self.depth = depth
#
#     def forward(self, obs):
#         batch_shape = obs.shape[:-3]
#         img_shape = obs.shape[-3:]
#         embed = self.convolutions(obs.reshape(-1, *img_shape) / 255. - 0.5)
#         embed = torch.reshape(embed, (*batch_shape, -1))
#         return embed
#
#     def spatial_attention(self, obs):
#         spatial_softmax = nn.Softmax(1)
#         img_shape = obs.shape[-3:]
#         gs = [None] * len(self.convolutions)
#         x = obs
#         for idx, layer in enumerate(self.convolutions):
#             x = layer(
#                 x.reshape(-1, *img_shape) / 255. - 0.5
#             ) if idx == 0 else layer(x)
#             gs[idx] = x
#         gs = [gs_.pow(2).mean(1) for gs_ in gs]
#         return [spatial_softmax(
#             gs_.view(*gs_.size()[:1], -1)
#         ).view_as(gs_) for gs_ in gs]
#
#     @property
#     def feature_dim(self):
#         conv1_shape = conv_out_shape(self.shape[1:], 0, 4, self.stride)
#         conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
#         conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
#         conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
#         embed_size = 8 * self.depth * np.prod(conv4_shape).item()
#         return embed_size


class CarlaObservationEncoder(nn.Module):
    def __init__(
        self, depth=32, stride=1, shape=(3, 64, 64), output_logits=False,
        num_layers=2, feature_dim=50, activation=nn.ReLU
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        out_dims = 100  # 5 cameras
        # outdims = 56  # 3 cameras
        self.embed_size = 256 * out_dims

        self.fc = nn.Linear(self.embed_size, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.shape = shape
        self.stride = stride
        self.depth = depth
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.output_logits = output_logits

    def forward_conv(self, obs):
        obs = obs / 255.

        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        return conv

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        obs = obs.reshape(-1, *img_shape)
        embed = self.forward_conv(obs)
        embed = torch.reshape(embed, (np.prod(batch_shape), -1))
        embed = self.fc(embed)
        embed = self.ln(embed)
        if not self.output_logits:
            embed = torch.tanh(embed)
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed

    def spatial_attention(self, obs):
        spatial_softmax = nn.Softmax(1)
        img_shape = obs.shape[-3:]
        gs = [None] * len(self.convolutions)
        x = obs
        for idx, layer in enumerate(self.convs):
            x = layer(
                x.reshape(-1, *img_shape) / 255.
            ) if idx == 0 else layer(x)
            gs[idx] = x
        gs = [gs_.pow(2).mean(1) for gs_ in gs]
        return [spatial_softmax(
            gs_.view(*gs_.size()[:1], -1)
        ).view_as(gs_) for gs_ in gs]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0], out_channels=self._out_channels,
            kernel_size=3, padding=1
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ImpalaEncoder(nn.Module):
    def __init__(self, shape=(3, 64, 64)):
        super().__init__()
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.shape = shape
        self.conv_seqs = nn.ModuleList(conv_seqs)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = obs.reshape(-1, *img_shape) / 255.
        for conv_seq in self.conv_seqs:
            embed = conv_seq(embed)
        embed = F.relu(embed)
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed

    def spatial_attention(self, obs):
        spatial_softmax = nn.Softmax(1)
        img_shape = obs.shape[-3:]
        gs = [None] * len(self.conv_seqs)
        x = obs
        for idx, layer in enumerate(self.conv_seqs):
            x = layer(
                x.reshape(-1, *img_shape) / 255.
            ) if idx == 0 else layer(x)
            gs[idx] = x
        gs = [gs_.pow(2).mean(1) for gs_ in gs]
        return [spatial_softmax(
            gs_.view(*gs_.size()[:1], -1)
        ).view_as(gs_) for gs_ in gs]

    @property
    def embed_size(self):
        embed_size = np.prod(self.shape)
        return embed_size


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)
