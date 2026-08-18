import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from MultiBuildingRLEnv.models.consumer_models import (
    RealConsumerModel,
    PredictedConsumerModel,
)
from MultiBuildingRLEnv.models.gen_models import (
    ClusterRepeatingGenerator,
    RepeatingGenerator,
)
from typing import Union
from stable_baselines3.common.callbacks import BaseCallback


class ClusterTrainingEnvWithGlobalCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        try:
            info_dict = self.locals["infos"][0]
            if self.locals["dones"][0]:
                for k in ["term_cons", "term_global", "cons_ref_over_agent"]:
                    self.logger.record(f"custom/{k}", info_dict[k])
        except Exception as e:
            print(e)
            print(self.locals)
            return False
        return True


class ClusterTrainingEnvWithGlobal(gym.Env):
    def __init__(
        self,
        cons_models: list[list[Union[RealConsumerModel, PredictedConsumerModel]]],
        load_generators: ClusterRepeatingGenerator,
        global_load_generator: RepeatingGenerator,
        weather_generator: list[RepeatingGenerator],
        n_prev_load: int = 2,
        n_prev_weather: int = 2,
        consumer_prices_min_max_tuple: tuple[float, float] = (0.2, 0.3),
        grid_cost_factor: float = 0.007,
        cons_factor_offset: float = 0.02,
    ) -> None:
        super().__init__()
        self.cms = cons_models
        self.lgs = load_generators  # a matrix of the type [[m11,m12,m13], ..mxy.. ] where mxy represents yth candidate model for xth consumer
        self.glg = global_load_generator
        self.tambg, self.qirrg = weather_generator
        self.grid_cost_factor = grid_cost_factor
        self.cons_factor_offset = cons_factor_offset
        # The observation and action spaces are to be defined here
        self.n_prev_load, self.n_prev_tamb = n_prev_load, n_prev_weather
        # time, n prev load, n_prev_grid_load, n prev weather
        space_mins = np.array(
            [0.0]
            + [-5.0] * n_prev_load
            + [-5.0] * n_prev_load
            + [-5.0] * n_prev_weather
        )
        space_maxs = np.array(
            [1.0] + [5.0] * n_prev_load + [5.0] * n_prev_load + [5.0] * n_prev_weather
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
        # Select a consumer model for this episode
        self.lgs.reset()  # This gives a new building index
        self.now_cm_idx = np.random.randint(0, len(self.cms[self.lgs.b_ix]))
        self.cm = self.cms[self.lgs.b_ix][
            self.now_cm_idx
        ]  # Pick one of the candidate models for this building
        # State tracking related variables:
        state = np.zeros_like(self.observation_space.sample())
        self.t_ix = 0
        self.prev_loads = [0.0] * self.n_prev_load
        self.prev_grid_loads = [0.0] * self.n_prev_load
        self.prev_tambs = [0.0] * self.n_prev_tamb
        # Reset the data generators and models
        day_idx = np.random.randint(0, len(self.glg.seq) // 96) * 96
        self.ref_par, self.ref_sum, self.ref_global = self._calc_ref_metrics(day_idx)
        self.act_prev_loads = []
        self.act_prev_grid_loads = []
        # For test-time state normalization
        self.test_state = state
        return state, {}

    def get_next_state(
        self, t: int, pi: float, grid_pow: float, net_grid_pow: float, t_amb: float
    ):
        """This method is intended for use from an external test interface. For the clustered agent case, assume that the input data
        is normalized."""
        # Normalize the observations
        n_load_n = grid_pow  # self.lg.normalize_sample(grid_pow)
        t_amb_n = t_amb  # self.tambg.normalize_sample(t_amb)
        n_load_grid = net_grid_pow  # self.glg.normalize_sample(net_grid_pow)
        # So we need to update the state
        self.prev_loads = self.prev_loads[1:] + [n_load_n]
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        self.prev_grid_loads = self.prev_grid_loads[1:] + [n_load_grid]
        next_state = np.array(
            [t / 24.0] + self.prev_loads + self.prev_grid_loads + self.prev_tambs
        )
        return next_state

    def _calc_ref_metrics(self, day_idx):
        # Reset the data generators and models
        self.cm.reset()
        self.glg.ix = day_idx
        self.lgs.t_ix = day_idx
        self.tambg.ix = day_idx
        self.qirrg.ix = day_idx
        # Run a simulation for this day with baseline pricing
        nl = []
        gl = []
        for t in range(96):
            e_load = self.lgs.get()
            grid_load = self.glg.get() - e_load
            t_amb = self.tambg.get()
            q_irr = self.qirrg.get()
            nl.append(self.cm.step(0.0, t_amb, q_irr, e_load))
            gl.append(grid_load)
        # Reset the data generators and models
        self.cm.reset()
        self.lgs.t_ix = day_idx
        self.glg.ix = day_idx
        self.tambg.ix = day_idx
        self.qirrg.ix = day_idx
        # Resample the net load profile into hourly
        nl_h = np.array(nl).reshape(24, -1).mean(axis=-1)
        par_nl_h = max(nl_h) / nl_h.mean()
        sum_nl_h = sum(nl_h)
        glob_ref = self.calc_global_term(
            np.array(nl), np.array(gl), offset_glob_ref=False
        )
        return par_nl_h, sum_nl_h, glob_ref

    def step(self, action):
        # One price should be used for 4 time steps
        pi = action.item()
        sub_loads = []
        sub_global_loads = []
        sub_temps = []
        for sub_ix in range(4):
            e_load = self.lgs.get()
            grid_load = (
                self.glg.get() - e_load
            )  # This is the residual load excluding the present building
            t_amb = self.tambg.get()
            q_irr = self.qirrg.get()
            sub_loads.append(self.cm.step(pi, t_amb, q_irr, e_load))
            sub_global_loads.append(grid_load)
            sub_temps.append(t_amb)
        n_load = sum(sub_loads) / 4  # average power in the last hour
        n_load_grid = sum(sub_global_loads) / 4  # average global power in the last hour
        t_amb = sum(sub_temps) / 4  # average temperature in the last hour
        # Normalize the observations
        n_load_n = self.lgs.normalize_sample(n_load)
        t_amb_n = self.tambg.normalize_sample(t_amb)
        n_load_grid = self.glg.normalize_sample(n_load_grid)
        # So we need to update the state
        self.t_ix += 1
        self.prev_loads = self.prev_loads[1:] + [n_load_n]
        self.prev_grid_loads = self.prev_grid_loads[1:] + [n_load_grid]
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        next_state = np.array(
            [self.t_ix / 24.0]
            + self.prev_loads
            + self.prev_grid_loads
            + self.prev_tambs
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
            # In both above cases, higher metric value is better (lower consumption, lower par)
            term_par = 0.0  # par_metric**5
            term_cons = self.calc_cons_term(
                cons_metric
            )  # scaling for hopefully easier learning
            reward = term_par + term_cons
            # We also add an extra term based on the correlation between the local and the global profile
            local_load = np.array(self.act_prev_loads)
            global_load = np.array(self.act_prev_grid_loads)
            term_global = self.calc_global_term(local_load, global_load)
            reward += term_global
            info = {
                "term_cons": term_cons,
                "term_global": term_global,
                "cons_ref_over_agent": cons_metric,
            }
        return (
            next_state,
            reward,
            done,
            False,
            info,
        )

    def calc_cons_term(self, cons_metric):
        if not hasattr(self, "cons_term_lin_eqn"):
            self.cons_term_knee_pt_x = (
                1.584 - self.cons_factor_offset
            )  # Solving the expo eqn for y=-100
            self.cons_term_lin_eqn = np.poly1d(
                np.polyfit([self.cons_term_knee_pt_x, 3.0], [-100, -300], deg=4)
            )
            self.cons_term_pos_lin = np.poly1d(
                np.polyfit([0.0, 1.0 - self.cons_factor_offset], [100, 1], deg=1)
            )
        x = (1 / cons_metric) + self.cons_factor_offset
        if x < 1.0 - self.cons_factor_offset:
            return self.cons_term_pos_lin(x)
        elif x < self.cons_term_knee_pt_x:
            # Exponential region
            return -(x**10)
        else:
            # Linear region
            return self.cons_term_lin_eqn(x)

    def calc_global_term(self, local_load, global_load, offset_glob_ref=True):
        corr = self.calc_correlation(local_load, global_load)
        term_global = 25 * np.tan(-0.5 * np.pi * corr)
        if offset_glob_ref:
            return term_global - self.ref_global
        return term_global

    def calc_correlation(self, actual, predic):
        a_diff = actual - np.mean(actual)
        p_diff = predic - np.mean(predic)
        numerator = np.sum(a_diff * p_diff)
        denominator = np.sqrt(np.sum(a_diff**2)) * np.sqrt(np.sum(p_diff**2))
        return numerator / denominator
