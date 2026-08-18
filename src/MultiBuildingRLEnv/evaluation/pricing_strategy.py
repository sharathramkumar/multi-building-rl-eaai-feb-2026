# Implementations for different pricing strategies
from abc import ABC, abstractmethod
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import os, pickle
from MultiBuildingRLEnv.envs.rl_envs import (
    SingleConsTrainingEnv,
    MultiConsSingleRLTrainingEnv,
)
from MultiBuildingRLEnv.envs.rl_envs_dev import (
    MultiConsSingleRLTrainingEnvWithFeedback,
    SingleConsTrainingEnvWithGlobal,
    SingleConsTrainingEnvWithGlobalFullState,
)
from MultiBuildingRLEnv.models.consumer_models import RealConsumerModel
from scipy.optimize import minimize
from d3rlpy.algos import BC, CalQL
from sklearn.preprocessing import MinMaxScaler
from typing import Union
import torch


class BasePricingStrategy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        # This signature is expected so that a "test" suite can be written to make use of it
        pass


class RandomPricingStrategy(BasePricingStrategy):
    def __init__(self):
        super().__init__()

    def __call__(self, t, grid_powers, t_amb, q_irrad, info_dict=None):
        return np.random.uniform(-1, 1, len(grid_powers))


class FixedLoadBasedPricingStrategy(BasePricingStrategy):
    def __init__(
        self,
        cons_models: list[RealConsumerModel],
        frac_load_threshold: float,
        aggregate: bool,
    ) -> None:
        super().__init__()
        self.cms = cons_models
        self.frac = frac_load_threshold
        self.t_ix = 0
        self.agg = aggregate
        avg_hourly_loads = []
        self.price_seqs = []
        for cm in self.cms:
            avg_hourly_load = (
                np.array(cm.net_load).reshape(-1, 96).reshape(-1, 24, 4).mean(axis=-1)
            )  # results in an array of shape (N_test_days, 24)
            avg_hourly_hvac_load = (
                np.array(cm.th_mod.history.p_ac)
                .reshape(-1, 96)
                .reshape(-1, 24, 4)
                .mean(axis=-1)
            )
            avg_hourly_loads.append(
                avg_hourly_load - avg_hourly_hvac_load
            )  # Save the fixed load matrix
        self.avg_hourly_loads = np.array(avg_hourly_loads)
        # print(np.array(self.avg_hourly_loads).shape) 5, 30, 24
        if self.agg:
            agg_fixed_load = self.avg_hourly_loads.sum(axis=0)
            agg_prices = []
            for day_load in agg_fixed_load:
                this_price_seq = self._get_one_price_seq(day_load)
                agg_prices += this_price_seq.tolist()
            self.price_seqs = (
                np.array(agg_prices * len(self.cms)).reshape(len(self.cms), -1).tolist()
            )
        else:
            for cons_load in self.avg_hourly_loads:
                this_cons_prices = []
                for day_load in cons_load:
                    this_price_seq = self._get_one_price_seq(day_load)
                    this_cons_prices += this_price_seq.tolist()
                self.price_seqs.append(this_cons_prices)

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        if (t % 4) == 0:
            self.prix = [cm_p[self.t_ix] for cm_p in self.price_seqs]
            self.t_ix += 1
        return self.prix

    def _get_one_price_seq(self, seq):
        # Take a 24-hour sequence and generate the price sequence
        this_price_sequence = np.zeros(
            24,
        )
        seq_sorted_inds = sorted(
            np.argsort(seq)[::-1][0 : int(self.frac * len(seq))]
        )  # This takes the top "N" indices
        this_price_sequence[seq_sorted_inds] = 1.0  # Assign high prices
        tx_checked = []
        for tx in seq_sorted_inds:
            if tx in tx_checked:
                continue
            ct = 0
            while tx + ct in seq_sorted_inds:
                tx_checked.append(tx + ct)
                ct += 1
            this_price_sequence[max(0, tx - (2 * ct)) : tx] = -0.5
        return this_price_sequence


class FutureKnownPricingStrategy(BasePricingStrategy):
    def __init__(
        self,
        cons_models: list[RealConsumerModel],
        n_days_lag: int = 0,
        use_fixed_loads: bool = False,
        responses: np.array = None,
        stagger_prices: bool = False,
    ) -> None:
        """
        Pricing Strategy that is derived from perfect knowledge of the future load under
        a different pricing strategy.

        Args:
            cons_models (list[RealConsumerModel]): List of consumer models whose net_load attribute contains simulation results.
        """
        super().__init__()
        self.cms = cons_models
        self.ref_prix = np.linspace(-1, 1, 24)
        self.price_seqs = []
        self.t_ix = 0
        self.n_days_lag = n_days_lag
        self.responses = responses
        for cm in self.cms:
            this_cm_prix = []
            avg_hourly_load = (
                np.array(cm.net_load).reshape(-1, 96).reshape(-1, 24, 4).mean(axis=-1)
            )  # results in an array of shape (N_test_days, 24)
            if use_fixed_loads:
                avg_hourly_hvac_load = (
                    np.array(cm.th_mod.history.p_ac)
                    .reshape(-1, 96)
                    .reshape(-1, 24, 4)
                    .mean(axis=-1)
                )
                avg_hourly_load = avg_hourly_load - avg_hourly_hvac_load
            for ahl in avg_hourly_load:
                sub_res = [0.0] * 24
                for i, v in enumerate(np.argsort(ahl)):
                    sub_res[v] = self.ref_prix[
                        i
                    ]  # redistribute the prices to match the load
                this_cm_prix += sub_res
            self.price_seqs.append(this_cm_prix)
        if self.n_days_lag > 0:
            for ix in range(len(self.price_seqs)):
                pad_len = self.n_days_lag * 24
                padding = [0.0] * pad_len
                self.price_seqs[ix] = padding + self.price_seqs[ix][0:-pad_len]
        if self.responses is not None:
            self._adjust_prices_to_responses(stagger_prices)

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        if (t % 4) == 0:
            self.prix = [cm_p[self.t_ix] for cm_p in self.price_seqs]
            self.t_ix += 1
        return self.prix

    def _adjust_prices_to_responses(self, stagger):
        # 1. Collect the responses matrix
        responses_h = self.responses.reshape(len(self.cms), -1, 4).mean(axis=-1)
        # 2. Filter out the periods where there was no response
        no_response = sum(responses_h.sum(axis=0) > 1.0)
        s = np.argsort(responses_h.sum(axis=0))[::-1][:no_response]
        self.price_seqs = self._update_sequences(
            np.array(self.price_seqs), self.responses, no_resp_indices=s, shift=stagger
        )

    def _update_sequences(self, prices, responses, no_resp_indices, shift=False):
        for nri in no_resp_indices:
            ms = prices[:, (nri - 2) : (nri + 3)].mean(axis=1)
            sh = -1
            for ix, m in enumerate(ms):
                if responses[ix, nri]:
                    sh += 1
                    n, p = self._get_hi_lo_prices(m)
                    if not shift:
                        prices[ix, (nri - 2) : (nri + 3)] = [n, n, p, p, p]
                    else:
                        prices[ix, (nri - 2) : (nri + 3)] = np.roll(
                            [n, n, p, p, p], shift=sh
                        )
        return prices

    def _get_hi_lo_prices(self, m):
        def objective_function(x, m):
            p, n = x
            return (2 * p + 3 * n - 5 * m) ** 2  # The equation you want to minimize

        # Initial guess for p and n
        initial_guess = [0.5, 0.5]

        # Define the bounds for p and n (positive real numbers less than 1)
        bounds = [(0.0, 0.2), (-1.0, -0.1)]

        # Run the optimization
        result = minimize(objective_function, initial_guess, bounds=bounds, args=(m,))

        # Extract the optimal values of p and n
        p_optimal, n_optimal = result.x

        return n_optimal, p_optimal


class RuleBasedPricingStrategy(BasePricingStrategy):
    def __init__(self):
        pass

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        return None


class StaticPricingStrategy(BasePricingStrategy):
    def __init__(self, level: float = 0.0) -> None:
        self.lev = float(np.clip(level, -1, 1))

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        return [self.lev for _x in range(len(grid_powers))]


class HourlyTimeOfUsePricingStrategy(BasePricingStrategy):
    def __init__(self, tou_seq: list) -> None:
        assert len(tou_seq) == 24, "Hourly TOU requires a 24-length price sequence!"
        self.seq = tou_seq

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        return [self.seq[t // 4] for _x in range(len(grid_powers))]


class IndividualRLPricingStrategy(BasePricingStrategy):
    def __init__(self, rl_models: list[PPO], rl_envs: list[SingleConsTrainingEnv]):
        """One RL agent for each house"""
        self.models = rl_models
        self.envs = rl_envs
        self.obss = []
        self.prix = [0.0] * len(rl_models)
        for env in self.envs:
            self.obss.append(env.reset()[0])
        # To track the observation update
        self.p_grid_prevs = np.zeros((len(rl_models), 4))
        self.t_amb_prevs = np.array([0.0] * 4)

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        # Update the buffers
        self.p_grid_prevs.T[(t % 4) - 1] = grid_powers
        self.t_amb_prevs[(t % 4) - 1] = t_amb
        # Check if we need a new price
        if (t % 4) == 0:
            # Get the price from each agent
            for ix, mod in enumerate(self.models):
                # Update the observations vector
                self.obss[ix] = self.envs[ix].get_next_state(
                    (t // 4) + 1,
                    self.prix[ix],
                    self.p_grid_prevs[ix].mean(),
                    self.t_amb_prevs.mean(),
                )
                # Get the new prices
                self.prix[ix] = mod.predict(self.obss[ix])[0].item()
        return self.prix


class IndividualRLPricingStrategyWithGlobal(BasePricingStrategy):
    def __init__(
        self,
        rl_models: list[PPO],
        rl_envs: list[SingleConsTrainingEnvWithGlobal],
        save_latent_pi_to: str = "",
    ):
        """One RL agent for each house"""
        self.models = rl_models
        self.envs = rl_envs
        self.obss = []
        self.prix = [0.0] * len(rl_models)
        for env in self.envs:
            self.obss.append(env.reset()[0])
        if os.path.isdir(save_latent_pi_to):
            self.save_latent_pi = True
            self.save_latent_pi_dir = save_latent_pi_to
        else:
            self.save_latent_pi = False
        # To track the observation update
        self.p_grid_prevs = np.zeros((len(rl_models), 4))
        self.t_amb_prevs = np.array([0.0] * 4)

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        # Update the buffers
        self.p_grid_prevs.T[(t % 4) - 1] = grid_powers
        self.t_amb_prevs[(t % 4) - 1] = t_amb
        # Check if we need a new price
        if (t % 4) == 0:
            # Get the price from each agent
            for ix, mod in enumerate(self.models):
                # Get the net load excluding this building
                net_load_excluding_ix = self.get_net_load_excluding_ix(ix)
                # Update the observations vector
                self.obss[ix] = self.envs[ix].get_next_state(
                    (t // 4) + 1,
                    self.prix[ix],
                    self.p_grid_prevs[ix].mean(),
                    net_load_excluding_ix,
                    self.t_amb_prevs.mean(),
                )
                # Get the new prices
                self.prix[ix] = mod.predict(self.obss[ix], deterministic=True)[0].item()
                if self.save_latent_pi:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(self.obss[ix].reshape(1, -1))
                        policy_features, value_features = mod.policy.extract_features(
                            obs_tensor
                        )
                        latent_pi = (
                            mod.policy.mlp_extractor.forward_actor(policy_features)
                            .detach()
                            .numpy()
                            .ravel()
                        )
                    with open(
                        os.path.join(self.save_latent_pi_dir, f"{ix}_lpi.dat"), "a+b"
                    ) as ff:
                        pickle.dump((self.obss[ix].ravel(), latent_pi), ff)
        return self.prix

    def get_net_load_excluding_ix(self, ix):
        inds = np.ones_like(self.p_grid_prevs, dtype=bool)
        inds[ix] = False
        return self.p_grid_prevs[inds].reshape(-1, 4).mean(axis=1).sum()


class IndividualRLPricingStrategyWithGlobalFullState(BasePricingStrategy):
    def __init__(
        self,
        rl_models: list[PPO],
        rl_envs: list[SingleConsTrainingEnvWithGlobalFullState],
        save_latent_pi_to: str = "",
    ):
        """One RL agent for each house"""
        self.models = rl_models
        self.envs = rl_envs
        self.n_calls = 0
        self.obss = []
        self.prix = [0.0] * len(rl_models)
        for env in self.envs:
            self.obss.append(env.reset()[0])
        if os.path.isdir(save_latent_pi_to):
            self.save_latent_pi = True
            self.save_latent_pi_dir = save_latent_pi_to
        else:
            self.save_latent_pi = False
        # To track the observation update
        self.p_grid_prevs = np.zeros((len(rl_models), 4))
        self.t_in_prevs = np.zeros((len(rl_models), 4))
        self.n_over_prevs = np.zeros((len(rl_models), 4))
        self.t_amb_prevs = np.array([0.0] * 4)

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,  # {"tps":test object}
    ) -> list[float]:
        # Update the buffers
        self.p_grid_prevs.T[(t % 4) - 1] = grid_powers
        self.t_amb_prevs[(t % 4) - 1] = t_amb
        if self.n_calls > 0:
            hidden_states = self.get_hidden_states(info_dict)
            self.t_in_prevs.T[(t % 4) - 1] = hidden_states["t_in"]
            self.n_over_prevs.T[(t % 4) - 1] = hidden_states["override"]
        self.n_calls += 1
        # Check if we need a new price
        if (t % 4) == 0:
            # Get the price from each agent
            for ix, mod in enumerate(self.models):
                # Get the net load excluding this building
                net_load_excluding_ix = self.get_net_load_excluding_ix(ix)
                # Update the observations vector
                self.obss[ix] = self.envs[ix].get_next_state(
                    (t // 4) + 1,
                    self.prix[ix],
                    self.p_grid_prevs[ix].mean(),
                    net_load_excluding_ix,
                    self.t_amb_prevs.mean(),
                    self.t_in_prevs[ix].mean(),
                    self.n_over_prevs[ix].mean(),
                )
                # Get the new prices
                self.prix[ix] = mod.predict(self.obss[ix], deterministic=True)[0].item()
                if self.save_latent_pi:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(self.obss[ix].reshape(1, -1))
                        policy_features, value_features = mod.policy.extract_features(
                            obs_tensor
                        )
                        latent_pi = (
                            mod.policy.mlp_extractor.forward_actor(policy_features)
                            .detach()
                            .numpy()
                            .ravel()
                        )
                    with open(
                        os.path.join(self.save_latent_pi_dir, f"{ix}_lpi.dat"), "a+b"
                    ) as ff:
                        pickle.dump((self.obss[ix].ravel(), latent_pi), ff)
        return self.prix

    def get_net_load_excluding_ix(self, ix):
        inds = np.ones_like(self.p_grid_prevs, dtype=bool)
        inds[ix] = False
        return self.p_grid_prevs[inds].reshape(-1, 4).mean(axis=1).sum()

    def get_hidden_states(self, info):
        # Gets the t_in and override vectors from the tps
        t_in = []
        override = []
        for cm in info["tps"].cms:
            tset = cm.spm.default_setpoint
            t_in.append(max(0, cm.th_mod.t_in - tset))
            override.append(cm.th_mod.ac.override_history[-1])
        return {"t_in": t_in, "override": override}


class MultiRLPricingStrategy(BasePricingStrategy):
    def __init__(self, rl_model: PPO, rl_env: MultiConsSingleRLTrainingEnv):
        """One RL agent for each house"""
        self.model = rl_model
        self.env = rl_env
        self.obs = self.env.reset()[0]
        self.prix = [0.0] * self.env.N
        # To track the observation update
        self.p_grid_prevs = np.zeros((self.env.N, 4))
        self.t_amb_prevs = np.array([0.0] * 4)

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        # Update the buffers
        self.p_grid_prevs.T[(t % 4) - 1] = grid_powers
        self.t_amb_prevs[(t % 4) - 1] = t_amb
        # Check if we need a new price
        if (t % 4) == 0:
            # Get the normalized observation from the agent
            self.obs = self.env.get_next_state(
                (t // 4) + 1,
                self.prix,
                self.p_grid_prevs.mean(axis=1).tolist(),
                self.t_amb_prevs.mean(),
            )
            self.prix = self.model.predict(self.obs)[0].tolist()
        return self.prix


class MultiRLPricingStrategyWithFeedback(BasePricingStrategy):
    def __init__(self, rl_model: PPO, rl_env: MultiConsSingleRLTrainingEnvWithFeedback):
        """One RL agent for each house"""
        self.model = rl_model
        self.env = rl_env
        self.obs = self.env.reset()[0]
        self.prix = [0.0] * self.env.N
        # To track the observation update
        self.p_grid_prevs = np.zeros((self.env.N, 4))
        self.n_over_prevs = np.zeros((self.env.N, 4))
        self.t_amb_prevs = np.array([0.0] * 4)
        self.n_calls = 0

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        self.n_calls += 1
        # Update the buffers
        self.p_grid_prevs.T[(t % 4) - 1] = grid_powers
        self.t_amb_prevs[(t % 4) - 1] = t_amb
        if self.n_calls > 1:
            self.n_over_prevs.T[(t % 4) - 1] = [
                cm.th_mod.ac.override_history[-1] for cm in self.env.cms
            ]
        # Check if we need a new price
        if (t % 4) == 0:
            # Get the normalized observation from the agent
            self.obs = self.env.get_next_state(
                (t // 4) + 1,
                self.prix,
                self.p_grid_prevs.mean(axis=1).tolist(),
                self.t_amb_prevs.mean(),
            )
            self.obs[len(self.obs) - self.env.N :] = self.n_over_prevs.mean(axis=1)
            self.prix = self.model.predict(self.obs)[0].tolist()
        return self.prix


class RecurrentMultiRLPricingStrategy(BasePricingStrategy):
    def __init__(self, rl_model: RecurrentPPO, rl_env: MultiConsSingleRLTrainingEnv):
        """One RL agent for each house"""
        self.model = rl_model
        self.env = rl_env
        self.obs = self.env.reset()[0]
        self.prix = [0.0] * self.env.N
        # To track the observation update
        self.p_grid_prevs = np.zeros((self.env.N, 4))
        self.t_amb_prevs = np.array([0.0] * 4)
        # LSTM states
        self.lstm_states = None

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        # Update the buffers
        self.p_grid_prevs.T[(t % 4) - 1] = grid_powers
        self.t_amb_prevs[(t % 4) - 1] = t_amb
        # Check if we need to reset the LSTM states (ie, start of new episode)
        if t == 0:
            self.lstm_states = None
        # Check if we need a new price
        if (t % 4) == 0:
            # Get the normalized observation from the agent
            self.obs = self.env.get_next_state(
                (t // 4) + 1,
                self.prix,
                self.p_grid_prevs.mean(axis=1).tolist(),
                self.t_amb_prevs.mean(),
            )
            act, self.lstm_states = self.model.predict(
                self.obs, state=self.lstm_states, deterministic=True
            )
            self.prix = act.tolist()
        return self.prix


class LoadBasedPWLPricingStrategy(BasePricingStrategy):
    def __init__(
        self, load_data: np.ndarray, d: float = 0.5, l0_d: float = 0.0
    ) -> None:
        self.ld = load_data  # array of dimension N, t_hist
        self.gl_mu, self.gl_std = self.ld.sum(axis=0).mean(), self.ld.sum(axis=0).std()
        ld_mets = np.array([self.ld.mean(axis=1), self.ld.std(axis=1)]).T
        self.ll_mus = ld_mets[:, 0]
        self.ll_sigs = ld_mets[:, 1]
        self.ll_hist = np.zeros((100, 8))
        self.gl_hist = np.zeros(8)
        self.ll_hh = np.tile(self.ll_mus, (4, 1)).T
        self.gl_hh = np.zeros(4) + self.gl_mu
        self.ww = np.ones(8)
        self.d = d
        self.gl_to_l0 = np.poly1d(
            np.polyfit([-5.0, 5.0], [5.0 + l0_d, -5.0 + l0_d], deg=1)
        )

    def get_pi(self):
        gl_m = np.dot(self.gl_hist, self.ww) / 8
        ll_m = np.dot(self.ll_hist, self.ww) / 8
        l0 = self.gl_to_l0(gl_m)
        lin_f = np.poly1d(np.polyfit([l0, l0 + self.d], [-1, 1], deg=1))
        return lin_f(ll_m).clip(-1, 1).tolist()

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        rllhh = np.roll(self.ll_hh, -1, axis=1)
        rllhh[:, -1] = grid_powers
        rglhh = np.roll(self.gl_hh, -1)
        rglhh[-1] = sum(grid_powers)
        self.ll_hh = rllhh
        self.gl_hh = rglhh
        if t % 4 == 0:
            gp_norm = (self.gl_hh.mean() - self.gl_mu) / self.gl_std
            ll_norms = (self.ll_hh.mean(axis=-1) - self.ll_mus) / self.ll_sigs
            rgp = np.roll(self.gl_hist, -1)
            rgp[-1] = gp_norm
            self.gl_hist = rgp
            rll = np.roll(self.ll_hist, -1, axis=1)
            rll[:, -1] = ll_norms
            self.ll_hist = rll
            self.last_prices = self.get_pi()
        return self.last_prices


class D3RLPyCommonRLPricingStrategy(BasePricingStrategy):
    def __init__(
        self,
        rl_model: Union[BC, CalQL],
        scalers: dict[str, Union[MinMaxScaler, dict[str, MinMaxScaler]]],
        ignore_ov: bool = False,
    ):
        self.model = rl_model
        self.scalers = scalers
        self.ignore_ov = ignore_ov
        self.n_cons = len(scalers.keys()) - 2
        n_inputs = 3 if ignore_ov else 4
        self.L = (
            self.model.observation_shape[0] // n_inputs
        )  # assuming tamb, pg, pgn, o_n as inputs
        self.obss = np.zeros((self.n_cons, self.model.observation_shape[0])) + 0.5
        if not ignore_ov:
            self.obss[
                :,
                self.model.observation_shape[0]
                - self.L : self.model.observation_shape[0],
            ] = 0  # Overrides are not minmax scaled
        self.prix = np.zeros(self.n_cons)
        # To track the observation update
        self.p_grid_prevs = np.zeros((self.n_cons, 4))
        self.overrides = np.zeros((self.n_cons, 4))
        self.t_amb_prevs = np.array([0.0] * 4)
        self.p_grid_agg_prevs = np.array([0.0] * 4)
        self.n_calls = 0

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        # Update the buffers
        self.p_grid_prevs.T[(t % 4) - 1] = grid_powers
        self.t_amb_prevs[(t % 4) - 1] = t_amb
        self.p_grid_agg_prevs[(t % 4) - 1] = sum(grid_powers)
        if self.n_calls > 0 and not self.ignore_ov:
            self.overrides.T[(t % 4) - 1] = self.get_hidden_states(info_dict)[
                "override"
            ]
        self.n_calls += 1
        # Check if we need a new price
        if (t % 4) == 0:
            self.update_obss()
            self.prix = self.model.predict(self.obss).flatten()
        return self.prix

    def update_obss(self):
        # Use the latest buffers to come up with the new state
        # Shift out the previous system state
        shifted_obss = np.roll(self.obss, -1, axis=1)
        new_tamb = self.scalers["t_amb"].transform([[self.t_amb_prevs.mean()]]).item()
        # print("DBG", "T amb", self.t_amb_prevs.mean(), "Scaled", new_tamb)
        shifted_obss[:, self.L - 1] = new_tamb
        new_agg_load = self.p_grid_agg_prevs.mean()
        # print("DBG", "Agg Ld", self.p_grid_agg_prevs.mean(), "Scaled", new_agg_load)
        for cix in range(self.n_cons):
            pg = self.p_grid_prevs[cix].mean()
            pgn = new_agg_load - pg
            new_load = self.scalers[f"cons_{cix}"]["pg"].transform([[pg]]).item()
            new_oth_load = self.scalers[f"cons_{cix}"]["pg_n"].transform([[pgn]]).item()
            shifted_obss[cix, 2 * self.L - 1] = new_load
            shifted_obss[cix, 3 * self.L - 1] = new_oth_load
            if not self.ignore_ov:
                new_overrides = self.overrides[cix].mean()
                shifted_obss[cix, 4 * self.L - 1] = new_overrides
        # print("DBG", "PG", pg, "Scaled", new_load)
        # print("DBG", "PGN", pgn, "Scaled", new_oth_load)
        # print("DBG", "OBSS", self.overrides[cix].mean())
        self.obss = shifted_obss

    def get_hidden_states(self, info):
        # Gets the t_in and override vectors from the tps
        t_in = []
        override = []
        for cm in info["tps"].cms:
            tset = cm.spm.default_setpoint
            t_in.append(max(0, cm.th_mod.t_in - tset))
            ov_hist = cm.th_mod.ac.override_history
            override.append(ov_hist[-1] if len(ov_hist) > 0 else 0)
        return {"t_in": t_in, "override": override}


class WeekdayBasedTOUStrategy(BasePricingStrategy):
    def __init__(self, tou_seqs: list[HourlyTimeOfUsePricingStrategy]) -> None:
        assert len(tou_seqs) == 7, "Expected one sequence for each day of the week!"
        self.tou_seqs = tou_seqs
        self.n_steps = 0

    def reset(self):
        self.n_steps = 0

    def __call__(
        self,
        t: int,
        grid_powers: list,
        t_amb: float,
        q_irrad: float,
        info_dict: dict = None,
    ) -> list[float]:
        tps = info_dict["tps"]
        dayofweek = tps.index[tps.start_idx + self.n_steps].weekday()
        self.n_steps += 1
        return self.tou_seqs[dayofweek](t, grid_powers, t_amb, q_irrad)


class HeuristicLoadBasedLinearStrategy(BasePricingStrategy):
    def __init__(
        self,
        init_loads: np.ndarray,
        lag_in_timesteps: int,
        transition_width_param: float = 6.0,
        ll_weight: float = 0.5,
        score_offset: float = 0.0,
    ):
        super().__init__()
        self.n_cons = init_loads.shape[
            0
        ]  # init_loads is of shape (n_cons, time_in_timesteps)
        self.loads = init_loads.reshape(self.n_cons, -1, 4).mean(
            axis=-1
        )  # (n_cons, time_in_hours) since 4 timesteps is 1 hour
        self.agg_loads: np.ndarray = self.loads.sum(axis=0)
        self.last_hour_load = init_loads[:, -4:]
        self.lag = lag_in_timesteps // 4
        self.twp = transition_width_param
        self.scores = np.zeros(self.n_cons)
        self.ll_weight = np.clip(ll_weight, 0.0, 1.0)
        self.offset = score_offset

    def __call__(self, t, grid_powers, t_amb, q_irrad, info_dict=None):
        self.last_hour_load.T[(t % 4) - 1] = grid_powers
        if (t % 4) == 0:
            # a new decision is needed
            new_data = self.last_hour_load.mean(axis=-1).reshape(-1, 1)
            self.loads = np.hstack((self.loads, new_data))
            self.agg_loads = self.loads.sum(axis=0)
            norm_ll, norm_gl = self.normalize()
            self.scores = (
                self.ll_weight * norm_ll + (1 - self.ll_weight) * norm_gl + self.offset
            ) / (0.5 * self.twp)
        return np.clip(self.scores.squeeze(), -1, 1)

    def normalize(self):
        # z-score normalization
        norm_ll_w = (
            self.loads[:, -self.lag :] - self.loads.mean(axis=1).reshape(-1, 1)
        ) / self.loads.std(axis=1).reshape(-1, 1)
        norm_gl_w = (
            self.agg_loads[-self.lag :] - self.agg_loads.mean()
        ) / self.agg_loads.std()
        norm_ll = norm_ll_w.mean(axis=1)
        norm_gl = norm_gl_w.mean()
        return norm_ll, norm_gl


class HeuristicAggLoadBasedRBC(BasePricingStrategy):
    def __init__(
        self,
        agg_load_threshold_kw: float,
        precool_periods: list[tuple] = [(16 * 4, 18 * 4)] * 5 + [(9 * 4, 11 * 4)] * 2,
        n_cons: int = 100,
        sig_limits: list[tuple] = [(0.1, 1.0), (-1.0, -0.1), (-0.3, 0.3)],
        n_groups: int = 10,
    ):
        super().__init__()
        self.threshold = agg_load_threshold_kw
        self.precool_periods = precool_periods
        self.load_hist = np.zeros((n_cons, 4))
        self.signals = np.zeros(n_cons)
        self.n = n_cons
        self.n_steps = 0
        self.sig_limits = sig_limits  # during dr, during precooling, otherwise
        self.n_groups = n_groups

    def __call__(self, t, grid_powers, t_amb, q_irrad, info_dict=None):
        self.load_hist.T[(t % 4) - 1] = grid_powers
        if (t % 4) == 0:
            # a new decision is needed
            tps = info_dict["tps"]
            dayofweek = tps.index[tps.start_idx + self.n_steps].weekday()
            is_pp = (t >= self.precool_periods[dayofweek][0]) & (
                t <= self.precool_periods[dayofweek][1]
            )
            last_hour_cons = self.load_hist.sum(axis=1)
            sorted_cons_idxs = np.argsort(last_hour_cons)
            grp_size = self.n // self.n_groups
            if sum(grid_powers) > (self.threshold * 1e3):
                # trigger a DR event
                sigs = np.linspace(
                    self.sig_limits[0][0], self.sig_limits[0][1], self.n_groups
                )
            elif is_pp:
                sigs = np.linspace(
                    self.sig_limits[1][0], self.sig_limits[1][1], self.n_groups
                )
            else:
                sigs = np.linspace(
                    self.sig_limits[2][0], self.sig_limits[2][1], self.n_groups
                )
            for i in range(self.n_groups):
                self.signals[sorted_cons_idxs[i * grp_size : (i + 1) * grp_size]] = (
                    sigs[i]
                )
        self.n_steps += 1
        return self.signals
