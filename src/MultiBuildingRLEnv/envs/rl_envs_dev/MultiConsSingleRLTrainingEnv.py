import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from MultiBuildingRLEnv.models.consumer_models import (
    RealConsumerModel,
    PredictedConsumerModel,
)
from MultiBuildingRLEnv.models.gen_models import BaseGenerator
from typing import Union


class MultiConsSingleRLTrainingEnv(gym.Env):
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
        )
        space_maxs = np.array(
            [1.0]  # time
            + [5.0] * n_prev_weather  # weather
            + [5.0] * n_prev_load * self.N  # prev loads for each consumer
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
        self.ep_net_load = [0.0] * 96  # assuming 96 steps per episode
        # Reset the data generators and models
        day_idx = np.random.randint(0, len(self.lgs[0].seq) // 96) * 96
        self.ref_peak_cons, self.ref_tot_cons = self._calc_ref_metrics(day_idx)
        # Reward tracking
        self.act_prev_loads = []
        # For test-time state normalization
        self.test_state = state
        return state, {}

    def _calc_ref_metrics(self, day_idx):
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
        nl = np.array(nl)
        tot_cons = nl.sum()
        peak_cons = (
            nl[nl > 1200 * len(self.cms)].sum() + 100
        )  # small offset to avoid divide-by-zero
        return peak_cons, tot_cons

    def step(self, action):
        # One price should be used for 4 time steps
        pi = action
        sub_loads = [[] for bix in range(self.N)]
        sub_temps = []
        for sub_ix in range(4):
            t_amb = self.tambg.get()
            q_irr = self.qirrg.get()
            for bix, (cm, lg) in enumerate(zip(self.cms, self.lgs)):
                e_load = lg.get()
                b_load = cm.step(pi[bix], t_amb, q_irr, e_load)
                sub_loads[bix].append(b_load)
                self.ep_net_load[(self.t_ix * 4) + sub_ix] += b_load
            sub_temps.append(t_amb)
        n_load = [sum(sl) / 4 for sl in sub_loads]  # average power in the last hour
        t_amb = sum(sub_temps) / 4  # average temperature in the last hour
        # Normalize the observations
        n_load_n = [lg.normalize_sample(n_load[bix]) for bix, lg in enumerate(self.lgs)]
        t_amb_n = self.tambg.normalize_sample(t_amb)
        # So we need to update the state
        self.t_ix += 1
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        for bix in range(self.N):
            self.prev_loads[bix] = self.prev_loads[bix][1:] + [n_load_n[bix]]
        next_state = [self.t_ix / 24.0] + self.prev_tambs
        for pl in self.prev_loads:
            next_state += pl
        next_state = np.array(next_state)
        # Calculate the reward
        if self.t_ix < 24:
            reward = 0.0
            done = False
        else:
            done = True
            this_ep_net_load = np.array(self.ep_net_load)
            this_ep_peak_load = this_ep_net_load[this_ep_net_load > 200_000].sum() + 100
            this_ep_total_load = this_ep_net_load.sum()
            peak_load_score = this_ep_peak_load / self.ref_peak_cons
            cons_score = this_ep_total_load / self.ref_tot_cons
            reward = self.calculate_reward(peak_load_score, cons_score)
        return next_state, reward, done, False, {}

    def calculate_reward(self, peak_load_score, cons_score):
        peak_term = -10 * np.log10(peak_load_score)
        cons_penalty = np.clip(1000 * np.log10(cons_score - 0.15), 0.0, 100.0)
        return peak_term - cons_penalty

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
        next_state = [self.t_ix / 24.0] + self.prev_tambs
        for pl in self.prev_loads:
            next_state += pl
        next_state = np.array(next_state)
        return next_state
