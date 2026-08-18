import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from MultiBuildingRLEnv.models.consumer_models import (
    RealConsumerModel,
    PredictedConsumerModel,
)
from MultiBuildingRLEnv.models.gen_models import BaseGenerator
from typing import Union


class MultiConsSingleRLTrainingEnvWithFeedback(gym.Env):
    def __init__(
        self,
        fitted_cons_models: list[Union[RealConsumerModel, PredictedConsumerModel]],
        load_generators: list[BaseGenerator],
        weather_generators: tuple[BaseGenerator, BaseGenerator],
        n_prev_load: int = 1,
        n_prev_weather: int = 1,
        consumer_prices_min_max_tuple: tuple[float, float] = (0.2, 0.3),
        grid_cost_factor: float = 0.007,
    ) -> None:
        """This environment is used to create a training environment for a single large neural network that accepts the inputs from all buildings simultaneously.

        Args:
            fitted_cons_models (list[Union[RealConsumerModel, PredictedConsumerModel]]): A list of fitted consumer models
            load_generators (list[BaseGenerator]): A list of load generators, one for each consumer
            weather_generators (tuple[BaseGenerator, BaseGenerator]): Weather generators, one for ambient temperature and one for solar irradiance.
            n_prev_load (int, optional): Number of previous load values to include in the state. Defaults to 2.
            n_prev_weather (int, optional): Number of previous temperature values to include in the state. Defaults to 2.
            consumer_prices_min_max_tuple (tuple[float, float], optional): Min and max retail electricity price for consumers in EUR. Defaults to (0.2, 0.3).
            grid_cost_factor (float, optional): Factor used to calculate the grid cost. Defaults to 0.007.
        """
        super().__init__()
        self.cms = fitted_cons_models
        self.lgs = load_generators
        self.tambg, self.qirrg = weather_generators
        self.grid_cost_factor = grid_cost_factor
        # The observation and action spaces are to be defined here
        self.n_prev_load, self.n_prev_tamb = n_prev_load, n_prev_weather
        self.N = len(fitted_cons_models)
        # The observation space is a concatenation of the observations for all the buildings in the cluster
        # time, n prev weather, n prev load * N, avg price * N, avg net load, peak net load
        space_mins = np.array(
            [0.0]  # time
            + [-5.0] * n_prev_weather  # weather
            + [-5.0] * n_prev_load * self.N  # prev loads for each consumer
            + [0.0]
            * self.N  # binary response status for each consumer, 0 means not overridden
            # + [-1.0] * self.N  # avg price for each consumer
            # + [-5.0, -10.0]  # avg and peak net load
        )
        space_maxs = np.array(
            [1.0]  # time
            + [5.0] * n_prev_weather  # weather
            + [5.0] * n_prev_load * self.N  # prev loads for each consumer
            + [1.0]
            * self.N  # binary response status for each consumer, 1 means was overridden
            # + [1.0] * self.N  # avg price for each consumer
            # + [5.0, 10.0]  # avg and peak net load
        )
        self.observation_space = Box(low=space_mins, high=space_maxs)
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.N,)
        )  # One action for each house
        self.act_to_price = np.poly1d(
            np.polyfit(
                [-1, 1],
                [consumer_prices_min_max_tuple[0], consumer_prices_min_max_tuple[1]],
                deg=1,
            )
        )

    def reset(self, seed=None):
        # State tracking related variables:
        state = np.zeros_like(self.observation_space.sample())
        self.t_ix = 0
        self.prev_loads = [[0.0] * self.n_prev_load for _b in range(self.N)]
        self.prev_tambs = [0.0] * self.n_prev_tamb
        self.prev_responses = [0.0] * self.N
        self.pi_avg = [0.0] * self.N
        self.n_load_avg = 0.0
        self.n_load_peak = -10.0
        # Reset the data generators and models
        day_idx = np.random.randint(0, len(self.lgs[0].seq) // 96) * 96
        self.ref_par = self._calc_ref_par(day_idx)
        # Reward tracking
        self.act_prev_loads = []
        self.act_prices = []
        self.bonus = 0.0
        # For test-time state normalization
        self.test_state = state
        return state, {}

    def _calc_ref_par(self, day_idx):
        # Reset the data generators and models
        for cm, lg in zip(self.cms, self.lgs):
            cm.reset()
            lg.ix = day_idx
        self.tambg.ix = day_idx
        self.qirrg.ix = day_idx
        # Run a simulation for this day with baseline pricing
        nl = []
        for t in range(96):
            t_amb = self.tambg.get()
            q_irr = self.qirrg.get()
            nl.append(0.0)
            for cm, lg in zip(self.cms, self.lgs):
                e_load = lg.get()
                this_tot_load = cm.step(0.0, t_amb, q_irr, e_load)
                nl[-1] += this_tot_load
        # Reset the data generators and models
        for cm, lg in zip(self.cms, self.lgs):
            cm.reset()
            lg.ix = day_idx
        self.tambg.ix = day_idx
        self.qirrg.ix = day_idx
        # Resample the net load profile into hourly
        nl_h = np.array(nl).reshape(24, -1).mean(axis=-1)
        return max(nl_h) / nl_h.mean()

    def step(self, action):
        # One price should be used for 4 time steps
        pi = action
        sub_loads = [[] for bix in range(self.N)]
        sub_overrides = [[] for bix in range(self.N)]
        sub_temps = []
        for sub_ix in range(4):
            t_amb = self.tambg.get()
            q_irr = self.qirrg.get()
            for bix, (cm, lg) in enumerate(zip(self.cms, self.lgs)):
                e_load = lg.get()
                b_load = cm.step(pi[bix], t_amb, q_irr, e_load)
                b_over = cm.th_mod.ac.override_history[-1]
                sub_loads[bix].append(b_load)
                sub_overrides[bix].append(b_over)
            sub_temps.append(t_amb)
        n_load = [sum(sl) / 4 for sl in sub_loads]  # average power in the last hour
        n_over = [
            sum(bo) / 4 for bo in sub_overrides
        ]  # avg number of overridden actions in the previous slot
        t_amb = sum(sub_temps) / 4  # average temperature in the last hour
        # Normalize the observations
        n_load_n = [lg.normalize_sample(n_load[bix]) for bix, lg in enumerate(self.lgs)]
        t_amb_n = self.tambg.normalize_sample(t_amb)
        # So we need to update the state
        self.t_ix += 1
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        for bix in range(self.N):
            self.prev_loads[bix] = self.prev_loads[bix][1:] + [n_load_n[bix]]
            self.pi_avg[bix] = ((self.pi_avg[bix] * (self.t_ix - 1)) + pi[bix]) / (
                self.t_ix
            )
        # A bit tricky to do - ideally, I would normalize the actual net load. Here, I assume that the
        # mean of the individual normalized building loads is a sufficient proxy
        n_load_avg_est = sum(n_load_n) / self.N
        self.n_load_avg = ((self.n_load_avg * (self.t_ix - 1)) + n_load_avg_est) / (
            self.t_ix
        )
        self.n_load_peak = (
            self.n_load_peak if self.n_load_peak > n_load_avg_est else n_load_avg_est
        )
        next_state = [self.t_ix / 24.0] + self.prev_tambs
        for pl in self.prev_loads:
            next_state += pl
        next_state += n_over
        # next_state += self.pi_avg + [self.n_load_avg] + [self.n_load_peak]
        next_state = np.array(next_state)
        # Update the reward tracking
        self.act_prev_loads.append(sum(n_load))
        self.act_prices.append(self.act_to_price(pi))
        # And calculate the reward
        expense = self.grid_cost_factor * ((sum(n_load) / 1000) ** 2)
        income = 0.0  # np.dot(self.act_prices[-1], n_load) / 1000
        profit = income - expense
        penalty = sum(n_over)
        # Each episode has 24 time steps
        # In this version the reward is strictly the profit
        reward = (5 * profit) - (10 * penalty)
        done = self.t_ix >= 24
        return next_state, reward, done, False, {}

    def get_next_state(
        self, t: int, pi: list[float], grid_pow: list[float], t_amb: float
    ):
        """This method is intended for use from an external test interface"""
        # Normalize the observations
        n_load_n = [
            lg.normalize_sample(grid_pow[bix]) for bix, lg in enumerate(self.lgs)
        ]
        t_amb_n = self.tambg.normalize_sample(t_amb)
        # So we need to update the state
        self.t_ix += 1
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        for bix in range(self.N):
            self.prev_loads[bix] = self.prev_loads[bix][1:] + [n_load_n[bix]]
            self.pi_avg[bix] = ((self.pi_avg[bix] * (self.t_ix - 1)) + pi[bix]) / (
                self.t_ix
            )
        # A bit tricky to do - ideally, I would normalize the actual net load. Here, I assume that the
        # mean of the individual normalized building loads is a sufficient proxy
        n_load_avg_est = sum(n_load_n) / self.N
        self.n_load_avg = ((self.n_load_avg * (self.t_ix - 1)) + n_load_avg_est) / (
            self.t_ix
        )
        self.n_load_peak = (
            self.n_load_peak if self.n_load_peak > n_load_avg_est else n_load_avg_est
        )
        next_state = [(self.t_ix % 24.0) / 24.0] + self.prev_tambs
        for pl in self.prev_loads:
            next_state += pl
        dummy_responses = [0.0] * self.N
        next_state += dummy_responses
        # next_state += self.pi_avg + [self.n_load_avg] + [self.n_load_peak]
        next_state = np.array(next_state)
        return next_state
