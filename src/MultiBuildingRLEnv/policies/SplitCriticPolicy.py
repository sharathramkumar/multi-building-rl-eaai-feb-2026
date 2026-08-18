from typing import Dict, List, Optional, Tuple, Type, Union, Callable
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution

import torch as th
import numpy as np
import torch.nn as nn
import gymnasium as gym
from functools import partial

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device


class FlattenAndStripNFeatures(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input and removes the bottom N features.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, n_to_strip: int = 0) -> None:
        self.n_features_in_obs = get_flattened_obs_dim(observation_space)
        assert (
            n_to_strip < self.n_features_in_obs
        ), f"Cannot strip {n_to_strip} elements from obs vector of {self.n_features_in_obs} features!"
        super().__init__(observation_space, self.n_features_in_obs - n_to_strip)
        self.n_to_strip = n_to_strip
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return th.narrow(
            self.flatten(observations), 1, 0, self.n_features_in_obs - self.n_to_strip
        )


class SplitActorCriticMlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: Union[int, Dict[str, int]],
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        if isinstance(feature_dim, int):
            last_layer_dim_pi = feature_dim
            last_layer_dim_vf = feature_dim
        elif isinstance(feature_dim, dict):
            last_layer_dim_pi = feature_dim["pi"]
            last_layer_dim_vf = feature_dim["vf"]

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class SplitInputActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        n_features_to_strip: int = 0,
        *args,
        **kwargs,
    ):
        self.mlp_extractor_kwargs = {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }
        super(SplitInputActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # non-shared features extractors for the actor and the critic
        self.policy_features_extractor = FlattenAndStripNFeatures(
            observation_space, n_features_to_strip
        )
        self.value_features_extractor = FlattenExtractor(observation_space)

        self.features_dim = {
            "pi": self.policy_features_extractor.features_dim,
            "vf": self.value_features_extractor.features_dim,
        }
        # NOTE: if the 2 features dims are different, your mlp_extractor must be able
        # to acceppt such dict AND ALSO an int, because the mlp_extractor will be first
        # created with wrong features_dim (coming from wrong, default, feratures extractor) which is an int.
        # Furthermore, note that with 2 different features dims the mlp_extractor cannot have shared layers.

        delattr(self, "features_extractor")  # remove the shared features extractor
        delattr(self, "pi_features_extractor")
        delattr(self, "vf_features_extractor")

        # Disable orthogonal initialization (if you want, otherwise comment it)
        self.ortho_init = False

        # The super-constructor calls a '_build' method that creates the network and the optimizer.
        # The problem is that it does so using a default features extractor, and not the ones just created,
        # therefore we need to re-create the mlp_extractor and the optimizer
        # (that otherwise would have used obsolete features_dims and parameters).
        self._rebuild(lr_schedule)

    def _rebuild(self, lr_schedule: Schedule) -> None:
        """Re-creates the mlp_extractor and the optimizer for the model.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # action_net and value_net as created in the '_build' method are OK,
        # no need to recreate them.

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.policy_features_extractor: np.sqrt(2),
                self.value_features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitActorCriticMlpExtractor(
            feature_dim=self.features_dim, **self.mlp_extractor_kwargs
        )

    def extract_features(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the feature extractor(s)
        """
        assert (
            self.policy_features_extractor is not None
            and self.value_features_extractor is not None
        )
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        policy_features = self.policy_features_extractor(preprocessed_obs)
        value_features = self.value_features_extractor(preprocessed_obs)
        return policy_features, value_features

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features)
        latent_vf = self.mlp_extractor.forward_critic(value_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        try:
            latent_pi = self.mlp_extractor.forward_actor(policy_features)
            latent_vf = self.mlp_extractor.forward_critic(value_features)
            distribution = self._get_action_dist_from_latent(latent_pi)
        except Exception as e:
            print("Exception!", e)
            print(
                "Nans at",
                np.argwhere(np.isnan(obs.detach().numpy())),
                np.argwhere(np.isnan(latent_pi.detach().numpy())),
                np.argwhere(np.isnan(latent_vf.detach().numpy())),
            )
            exit(-1)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :return: the action distribution.
        """
        policy_features, _ = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, value_features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        return self.value_net(latent_vf)
