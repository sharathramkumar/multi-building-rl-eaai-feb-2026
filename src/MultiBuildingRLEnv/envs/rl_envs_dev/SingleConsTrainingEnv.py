import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class SingleConsTrainingEnv(gym.Env):
    def __init__(
        self,
        fitted_cons_model,
        load_generator,
        weather_generator,
        n_prev_load: int = 2,
        n_prev_weather: int = 2,
        consumer_prices_min_max_tuple: tuple[float, float] = (0.2, 0.3),
        grid_cost_factor: float = 0.007,
    ) -> None:
        """This training environment can be used to train a single RL agent for one type of fitted model.

        Args:
            fitted_cons_model (RealConsumerModel or PredictedConsumerModel): The consumer model with fitted parameters and a step function
            load_generator (RepeatingGenerator-like): The load generator model, which can be iteratively queried and returns the next load in W
            weather_generator (Tuple[Generator-like, Generator-like]): A tuple containing generators for T_amb and Q_irrad
            n_prev_load (int, optional): Number of previous load values to include in the state. Defaults to 2.
            n_prev_weather (int, optional): Number of previous temperature values to include in the state. Defaults to 2.
            consumer_prices_min_max_tuple (tuple[float, float], optional): Min and max retail electricity price for consumers in EUR. Defaults to (0.2, 0.3).
            grid_cost_factor (float, optional): Factor used to calculate the grid cost. Defaults to 0.007.
        """
        super().__init__()
        self.cm = fitted_cons_model
        self.lg = load_generator
        self.tambg, self.qirrg = weather_generator
        self.grid_cost_factor = grid_cost_factor
        # The observation and action spaces are to be defined here
        self.n_prev_load, self.n_prev_tamb = n_prev_load, n_prev_weather
        # time, n prev load, n prev weather, avg price, avg load, peak load
        space_mins = np.array(
            [0.0]
            + [-5.0] * n_prev_load
            + [-5.0] * n_prev_weather
            # + [-1.0]
            # + [-5.0]
            # + [-10.0]
        )
        space_maxs = np.array(
            [1.0]
            + [5.0] * n_prev_load
            + [5.0] * n_prev_weather
            # + [1.0]
            # + [5.0]
            # + [10.0]
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
        self.prev_tambs = [0.0] * self.n_prev_tamb
        self.pi_avg = 0.0
        self.n_load_avg = 0.0
        self.n_load_peak = -10.0
        # Reset the data generators and models
        day_idx = np.random.randint(0, len(self.lg.seq) // 96) * 96
        self.ref_par, self.ref_sum = self._calc_ref_par(day_idx)
        # Reward tracking
        self.act_prev_loads = []
        self.act_prices = []
        self.n_load_act_avg = 0.0
        self.bonus = 0.0
        # For test-time state normalization
        self.test_state = state
        return state, {}

    def get_next_state(self, t: int, pi: float, grid_pow: float, t_amb: float):
        """This method is intended for use from an external test interface"""
        # Normalize the observations
        n_load_n = self.lg.normalize_sample(grid_pow)
        t_amb_n = self.tambg.normalize_sample(t_amb)
        # So we need to update the state
        self.prev_loads = self.prev_loads[1:] + [n_load_n]
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        # self.pi_avg = ((self.pi_avg * (t - 1)) + pi) / (t)
        # self.n_load_avg = ((self.n_load_avg * (t - 1)) + n_load_n) / (t)
        # self.n_load_peak = self.n_load_peak if self.n_load_peak > n_load_n else n_load_n
        next_state = np.array(
            [t / 24.0]
            + self.prev_loads
            + self.prev_tambs
            # + [self.pi_avg]
            # + [self.n_load_avg]
            # + [self.n_load_peak]
        )
        return next_state

    def _calc_ref_par(self, day_idx):
        # Reset the data generators and models
        self.cm.reset()
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
        sub_temps = []
        for sub_ix in range(4):
            e_load = self.lg.get()
            t_amb = self.tambg.get()
            q_irr = self.qirrg.get()
            sub_loads.append(self.cm.step(pi, t_amb, q_irr, e_load))
            sub_temps.append(t_amb)
        n_load = sum(sub_loads) / 4  # average power in the last hour
        t_amb = sum(sub_temps) / 4  # average temperature in the last hour
        # Normalize the observations
        n_load_n = self.lg.normalize_sample(n_load)
        t_amb_n = self.tambg.normalize_sample(t_amb)
        # So we need to update the state
        self.t_ix += 1
        self.prev_loads = self.prev_loads[1:] + [n_load_n]
        self.prev_tambs = self.prev_tambs[1:] + [t_amb_n]
        # self.pi_avg = ((self.pi_avg * (self.t_ix - 1)) + pi) / (self.t_ix)
        # self.n_load_avg = ((self.n_load_avg * (self.t_ix - 1)) + n_load_n) / (self.t_ix)
        # self.n_load_peak = self.n_load_peak if self.n_load_peak > n_load_n else n_load_n
        # self.n_load_act_avg = ((self.n_load_act_avg * (self.t_ix - 1)) + n_load) / (
        #     self.t_ix
        # )
        next_state = np.array(
            [self.t_ix / 24.0]
            + self.prev_loads
            + self.prev_tambs
            # + [self.pi_avg]
            # + [self.n_load_avg]
            # + [self.n_load_peak]
        )
        # Update the reward tracking
        self.act_prev_loads.append(n_load)
        self.act_prices.append(self.act_to_price(pi))
        # And calculate the reward
        if self.t_ix < 24:
            reward = 0.0
            done = False
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
            reward += par_metric**5 - (
                (((1 / cons_metric) + 0.04) ** 10)
            )  # scaling for hopefully easier learning
        return next_state, reward, done, False, {}
