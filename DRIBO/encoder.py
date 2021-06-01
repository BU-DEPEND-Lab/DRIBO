import torch
import torch.nn as nn
import numpy as np
from DRIBO.rssm import RSSMState, RSSMRepresentation, RSSMTransition, \
    RSSMRollout, ObservationEncoder, CarlaObservationEncoder
from DRIBO import pytorch_util as ptu


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(
        self, obs_shape, feature_dim,
        num_layers=2, num_filters=32, output_logits=False
    ):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] \
            if obs_shape[-1] == 64 else OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class RSSMEncoder(nn.Module):
    """RSSM encoder of pixels observations."""
    def __init__(
        self, obs_shape, action_shape, feature_dim=50,
        stochastic_size=30, deterministic_size=200, num_layers=2,
        num_filters=32, hidden_size=200, dtype=torch.float, output_logits=False
    ):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.action_shape = np.prod(action_shape)
        action_size = np.prod(action_shape)
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.feature_dim = stochastic_size + deterministic_size
        self.num_layers = num_layers
        self.dtype = dtype
        self.output_logits = output_logits
        self.outputs = dict()

        # Pixel encoder
        self.observation_encoder = ObservationEncoder(
            shape=obs_shape, num_layers=num_layers, feature_dim=feature_dim,
            depth=num_filters, output_logits=self.output_logits
        )
        pixel_embed_size = self.observation_encoder.feature_dim

        # RSSM model
        self.transition = RSSMTransition(
            action_size, stochastic_size, deterministic_size, hidden_size
        )
        self.representation = RSSMRepresentation(
            self.transition, pixel_embed_size, action_size,
            stochastic_size, deterministic_size, hidden_size
        )
        self.rollout = RSSMRollout(self.representation, self.transition)

        # layer_norm
        self.ln = nn.LayerNorm(self.feature_dim)

    def get_state_representation(
        self, observation: torch.Tensor,
        prev_action: torch.Tensor = None, prev_state: RSSMState = None
    ):
        """
        :param observation: size(batch, channels, width, height)
        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        observation = observation
        obs_embed = self.observation_encoder(observation)
        if prev_action is None:
            prev_action = torch.zeros(
                observation.size(0), self.action_shape, device=ptu.device
            )
        if prev_state is None:
            prev_state = self.representation.initial_state(
                prev_action.size(0), device=ptu.device
            )
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state

    def forward(
        self, obs, prev_action=None,
        prev_state: RSSMState = None,
    ):
        state = self.get_state_representation(obs, prev_action, prev_state)
        return state

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)


class RSSMCarlaEncoder(nn.Module):
    """RSSM Carla encoder of pixels observations."""
    def __init__(
        self, obs_shape, action_shape, feature_dim=50,
        stochastic_size=30, deterministic_size=200, num_layers=2,
        num_filters=32, hidden_size=200, dtype=torch.float, output_logits=False
    ):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.action_shape = np.prod(action_shape)
        action_size = np.prod(action_shape)
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.feature_dim = stochastic_size + deterministic_size
        self.num_layers = num_layers
        self.dtype = dtype
        self.output_logits = output_logits
        self.outputs = dict()

        # Pixel encoder
        self.observation_encoder = CarlaObservationEncoder(
            shape=obs_shape, num_layers=num_layers, feature_dim=feature_dim,
            depth=num_filters, output_logits=self.output_logits
        )
        pixel_embed_size = self.observation_encoder.feature_dim

        # RSSM model
        self.transition = RSSMTransition(
            action_size, stochastic_size, deterministic_size, hidden_size
        )
        self.representation = RSSMRepresentation(
            self.transition, pixel_embed_size, action_size,
            stochastic_size, deterministic_size, hidden_size
        )
        self.rollout = RSSMRollout(self.representation, self.transition)

        # layer_norm
        self.ln = nn.LayerNorm(self.feature_dim)

    def get_state_representation(
        self, observation: torch.Tensor,
        prev_action: torch.Tensor = None, prev_state: RSSMState = None
    ):
        """
        :param observation: size(batch, channels, width, height)
        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        observation = observation
        obs_embed = self.observation_encoder(observation)
        if prev_action is None:
            prev_action = torch.zeros(
                observation.size(0), self.action_shape, device=ptu.device
            )
        if prev_state is None:
            prev_state = self.representation.initial_state(
                prev_action.size(0), device=ptu.device
            )
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state

    def forward(
        self, obs, prev_action=None,
        prev_state: RSSMState = None,
    ):
        state = self.get_state_representation(obs, prev_action, prev_state)
        return state

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)



_AVAILABLE_ENCODERS = {
    'pixel': PixelEncoder, 'identity': IdentityEncoder, 'rssm': RSSMEncoder,
    'carla_rssm': RSSMCarlaEncoder,
}


def make_encoder(
    encoder_type, obs_shape, feature_dim,
    num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )


def make_rssm_encoder(
    encoder_type, obs_shape, action_shape, feature_dim, stochastic_size=30,
    deterministic_size=200, num_layers=2, num_filters=32, hidden_size=200,
    dtype=torch.float, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, action_shape, feature_dim,
        stochastic_size, deterministic_size,
        num_layers, num_filters, hidden_size, dtype, output_logits
    )
