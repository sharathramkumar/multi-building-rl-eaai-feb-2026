import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from MultiBuildingRLEnv.models.consumer_models import (
    RealConsumerModel,
    PredictedConsumerModel,
)
from MultiBuildingRLEnv.models.gen_models import BaseGenerator
from typing import Union


class SingleConsTrainingEnvWithGlobalFullState(gym.Env):
    def __init__(
        self,
        fitted_cons_model: Union[RealConsumerModel, PredictedConsumerModel],
        load_generator: BaseGenerator,
        global_load_generator: BaseGenerator,
        weather_generator: list[BaseGenerator],
        n_prev_load: int = 2,
        n_prev_weather: int = 2,
        consumer_prices_min_max_tuple: tuple[float, float] = (0.2, 0.3),
        grid_cost_factor: float = 0.007,
        cons_factor_offset: float = 0.02,
    ) -> None:
        super().__init__()
        self.cm = fitted_cons_model
        self.lg = load_generator
        self.glg = global_load_generator
        self.tambg, self.qirrg = weather_generator
        self.grid_cost_factor = grid_cost_factor
        self.cons_factor_offset = cons_factor_offset
        self.tset = self.cm.spm.default_setpoint
        # The observation and action spaces are to be defined here
        self.n_prev_load, self.n_prev_tamb = n_prev_load, n_prev_weather
        # time, n prev load, n_prev_grid_load, n prev weather, n_prev_t_in, n_prev_overrides
        space_mins = np.array(
            [0.0]
            + [-5.0] * n_prev_load
            + [-5.0] * n_prev_load
            + [-5.0] * n_prev_weather
            + [0.0] * n_prev_weather
            + [0] * n_prev_weather
        )
        space_maxs = np.array(
            [1.0]
            + [5.0] * n_prev_load
            + [5.0] * n_prev_load
            + [5.0] * n_prev_weather
            + [3.0] * n_prev_weather
            + [1.0] * n_prev_weather
        )
        self.observation_space = Box(low=space_mins, high=space_maxs)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
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
        self.prev_loads = [0.0] * self.n_prev_load
        self.prev_grid_loads = [0.0] * self.n_prev_load
        self.prev_tambs = [0.0] * self.n_prev_tamb
        self.prev_tins = [0.0] * self.n_prev_tamb
        self.prev_overrides = [0.0] * self.n_prev_tamb
        # Reset the data generators and models
        day_idx = np.random.randint(0, len(self.lg.seq) // 96) * 96
        self.ref_par, self.ref_sum = self._calc_ref_par(day_idx)
        self.act_prev_loads = []
        self.act_prev_grid_loads = []
        # For test-time state normalization
        self.test_state = state
        return state, {}

    def get_next_state(
        self,
        t: int,
        pi: float,
        grid_pow: float,
        net_grid_pow: float,
        t_amb: float,
        dt_in: float,
        n_over: float,
        info_dict: dict = None,
    ):
        """This method is intended for use from an external test interface"""
        # Normalize the observations
        n_load_n = self.lg.normalize_sample(grid_pow)
        t_amb_n = self.tambg.normalize_sample(t_amb)
        n_load_grid = self.glg.normalize_sample(net_grid_pow)
        # So we need to update the state
        self.prev_loads = self.prev_loads[1:] + [n_load_n]
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        self.prev_grid_loads = self.prev_grid_loads[1:] + [n_load_grid]
        self.prev_tins = self.prev_tins[1:] + [dt_in]
        self.prev_overrides = self.prev_overrides[1:] + [n_over]
        next_state = np.array(
            [self.t_ix / 24.0]
            + self.prev_loads
            + self.prev_grid_loads
            + self.prev_tambs
            + self.prev_tins
            + self.prev_overrides
        )
        return next_state

    def _calc_ref_par(self, day_idx):
        # Reset the data generators and models
        self.cm.reset()
        self.glg.ix = day_idx
        self.lg.ix = day_idx
        self.tambg.ix = day_idx
        self.qirrg.ix = day_idx
        # Run a simulation for this day with baseline pricing
        nl = []
        for t in range(96):
            e_load = self.lg.get()
            t_amb = self.tambg.get()
            q_irr = self.qirrg.get()
            nl.append(self.cm.step(0.0, t_amb, q_irr, e_load))
        # Reset the data generators and models
        self.cm.reset()
        self.lg.ix = day_idx
        self.tambg.ix = day_idx
        self.qirrg.ix = day_idx
        # Resample the net load profile into hourly
        nl_h = np.array(nl).reshape(24, -1).mean(axis=-1)
        par_nl_h = max(nl_h) / nl_h.mean()
        sum_nl_h = sum(nl_h)
        return par_nl_h, sum_nl_h

    def step(self, action):
        # One price should be used for 4 time steps
        pi = action.item()
        sub_loads = []
        sub_global_loads = []
        sub_temps = []
        sub_tins = []
        sub_overrides = []
        for sub_ix in range(4):
            e_load = self.lg.get()
            grid_load = self.glg.get()
            t_amb = self.tambg.get()
            q_irr = self.qirrg.get()
            sub_loads.append(self.cm.step(pi, t_amb, q_irr, e_load))
            sub_global_loads.append(grid_load)
            sub_temps.append(t_amb)
            sub_tins.append(max(0, self.cm.th_mod.t_in - self.tset))
            sub_overrides.append(self.cm.th_mod.ac.override_history[-1])
        n_load = sum(sub_loads) / 4  # average power in the last hour
        n_load_grid = sum(sub_global_loads) / 4  # average global power in the last hour
        t_amb = sum(sub_temps) / 4  # average temperature in the last hour
        dt_in = sum(sub_tins) / 4  # average indoor temp deviation from setpoint
        n_over = (
            sum(sub_overrides) / 4
        )  # fraction of time commands were overridden in the previous slot
        # Normalize the observations
        n_load_n = self.lg.normalize_sample(n_load)
        t_amb_n = self.tambg.normalize_sample(t_amb)
        n_load_grid = self.glg.normalize_sample(n_load_grid)
        # So we need to update the state
        self.t_ix += 1
        self.prev_loads = self.prev_loads[1:] + [n_load_n]
        self.prev_grid_loads = self.prev_grid_loads[1:] + [n_load_grid]
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        self.prev_tins = self.prev_tins[1:] + [dt_in]
        self.prev_overrides = self.prev_overrides[1:] + [n_over]
        next_state = np.array(
            [self.t_ix / 24.0]
            + self.prev_loads
            + self.prev_grid_loads
            + self.prev_tambs
            + self.prev_tins
            + self.prev_overrides
        )
        # Update the reward tracking
        self.act_prev_loads.append(n_load)
        self.act_prev_grid_loads.append(n_load_grid)
        # And calculate the reward
        if self.t_ix < 24:
            reward = 0.0
            done = False
            info = {}
        else:
            done = True
            nl_sum = sum(self.act_prev_loads)
            par = max(self.act_prev_loads) / (
                sum(self.act_prev_loads) / len(self.act_prev_loads)
            )
            reward = 0.0
            # We choose a simple reward formulation for exactly the criteria we would like to optimize
            cons_metric = self.ref_sum / nl_sum
            par_metric = self.ref_par / par
            # In both above cases, higher metric value is better (lower consumption, lower par)
            term_par = 0.0  # par_metric**5
            term_cons = -(
                (((1 / cons_metric) + self.cons_factor_offset) ** 10)
            )  # scaling for hopefully easier learning
            reward = term_par + term_cons
            # We also add an extra term based on the correlation between the local and the global profile
            local_load = np.array(self.act_prev_loads)
            global_load = np.array(self.act_prev_grid_loads)
            corr = self.calc_correlation(local_load, global_load)
            term_global = 10 * np.tan(-0.5 * np.pi * corr)
            reward += term_global
            info = {
                "term_par": term_par,
                "term_cons": term_cons,
                "term_global": term_global,
            }
        return (
            next_state,
            reward,
            done,
            False,
            info,
        )

    def calc_correlation(self, actual, predic):
        a_diff = actual - np.mean(actual)
        p_diff = predic - np.mean(predic)
        numerator = np.sum(a_diff * p_diff)
        denominator = np.sqrt(np.sum(a_diff**2)) * np.sqrt(np.sum(p_diff**2))
        return numerator / denominator
