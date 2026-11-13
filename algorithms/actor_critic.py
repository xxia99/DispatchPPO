import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.encoder import EncoderBase as EncoderBase
from algorithms.utils.rnn import RNNLayer
# from algorithms.utils.act import ACTLayer
from algorithms.utils.decoder import ACTLayer
from algorithms.utils.util import get_shape_from_obs_space


class Actor(nn.Module):
    """
    DispatchNet actor: CNN grid features, Transformer encoder over the UAV tokens, per-UAV move decoder.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.args=args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.epsilon = args.explore_eps
        self.epsilon_ic = args.epsilon_ic
        # print("Using device:", device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = CNNBase(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.grid_size = obs_shape[-1]
        # self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        self.encoder = EncoderBase(args, obs_shape)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain,
                            device=device, grid_size=self.grid_size, args=args)

        self.to(device)

    def set_epsilon(self, epsilon: float):
        """Set exploration epsilon used for epsilon-greedy action selection."""
        self.epsilon = epsilon

    def forward(self, obs, rnn_states, masks, uav_pos, available_actions=None, deterministic=False, force_m=False, m=None, adj=None, epsilon=None, execution=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # print("m:", m)
        obs = check(obs).to(**self.tpdv)
        # print("obs:", obs.shape, obs.dtype)
        w_map = obs[:, 0, :, :] if execution else None
        uav_positions = check(uav_pos).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            _, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actor_features = self.encoder(actor_features, uav_positions)
        # actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        eps = self.epsilon if epsilon is None else epsilon
        if deterministic:
            eps = 0.0
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic,
                                             uav_positions=uav_positions,
                                             epsilon=eps,
                                             epsilon_ic=self.epsilon_ic,
                                             w_map=w_map)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, uav_pos,
        available_actions=None, active_masks=None, adj=None, epsilon=None, execution=False):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        w_map = obs[:, 0, :, :] if execution else None
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        # print("uav_pos:", uav_pos)
        uav_pos = check(uav_pos).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.args.use_gnn:
            if adj is None:
                bs = obs.shape[0]
                n = obs.shape[1]
                adj = torch.eye(n, device=obs.device).unsqueeze(0).repeat(bs,1,1)
            actor_features = self.base(obs, adj)
            actor_features = actor_features.view(-1, actor_features.shape[-1])
        else:
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            _, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actor_features = self.encoder(actor_features, uav_pos)
        eps = self.epsilon if epsilon is None else epsilon
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                action, available_actions,
                                                                active_masks=
                                                                active_masks if self._use_policy_active_masks
                                                                else None,
                                                                uav_positions=uav_pos,
                                                                epsilon=eps,
                                                                epsilon_ic=self.epsilon_ic,
                                                                w_map=w_map)

        return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    Value network over the centralized observation.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.base = CNNBase(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, uav_positions=None):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
