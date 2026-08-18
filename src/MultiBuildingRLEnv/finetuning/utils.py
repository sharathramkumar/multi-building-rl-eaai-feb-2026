import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Union
from MultiBuildingRLEnv.envs.rl_envs_dev import SingleConsTrainingEnvWithGlobalFullState
from MultiBuildingRLEnv.evaluation.tests import TestPricingStrategy
from MultiBuildingRLEnv.models.consumer_models import RealConsumerModel
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def downsample(series, factor=4):
    """
    Downsamples a 1D array by averaging over groups of `factor`.
    """
    return series[: len(series) // factor * factor].reshape(-1, factor).mean(axis=1)


def prepare_replay_buffer(
    pg, pg_n, o_n, t_amb_scaled, a_t, L, scalers: dict[str, MinMaxScaler] = None
):
    """
    Prepares data for a replay buffer by scaling, slicing, and pairing states with actions.

    Args:
        pg (np.array): Time series of the first state variable with shape (T,).
        pg_n (np.array): Time series of the second state variable with shape (T,).
        o_n (np.array): Time series of the third state variable with shape (T,).
        t_amb_scaled (np.array): Scaled time series of the fourth state variable with shape (downsampled_T,).
        a_t (np.array): Time series of actions with shape (T,).
        L (int): Length of each slice (window).
        scalers : Dict of MinMaxScalers

    Returns:
        states (np.array): Array of sliced states with shape (num_windows, L, 3).
        actions (np.array): Array of actions corresponding to each slice with shape (num_windows, 1).
        dones (np.array): Array of dones corresponding to each slice with shape (num_windows, 1).
        scalers (dict): Dictionary containing MinMaxScaler objects for each state variable.
    """

    # Step 0: Downsampling
    pg = downsample(pg)
    pg_n = downsample(pg_n)
    o_n = downsample(o_n)
    a_t = downsample(a_t)

    # Step 1: MinMax scaling
    if scalers is None:
        scalers = {
            "pg": MinMaxScaler(),
            "pg_n": MinMaxScaler(),
        }
        pg_scaled = (
            scalers["pg"].fit_transform(pg.reshape(-1, 1)).flatten()
        )  # Reshape for single feature
        pg_n_scaled = scalers["pg_n"].fit_transform(pg_n.reshape(-1, 1)).flatten()
    else:
        # Reuse the existing scaler
        pg_scaled = (
            scalers["pg"].transform(pg.reshape(-1, 1)).flatten()
        )  # Reshape for single feature
        pg_n_scaled = scalers["pg_n"].transform(pg_n.reshape(-1, 1)).flatten()

    # Combine the scaled states into a single array
    for x in [t_amb_scaled, pg_scaled, pg_n_scaled, o_n]:
        logger.debug(f"shape= {x.shape}")
    states_combined = np.stack(
        [t_amb_scaled, pg_scaled, pg_n_scaled, o_n], axis=1
    )  # Shape: (T, 4)
    # print("shape of combined states", states_combined.shape)

    # Step 2: Slice the inputs and pair with actions
    states, actions, dones = [], [], []
    for i in range(len(pg) - L - 1):
        # Create a window of states
        states.append(states_combined[i : i + L].T.ravel())  # Shape: (L*4,)

        # The action corresponding to the window is at position i + L
        actions.append(a_t[i + L + 1])  # Single action

        # The done flag for this state
        if (i + L + 1) % 24 == 0:
            dones.append(True)
        else:
            dones.append(False)

    # Convert to numpy arrays
    states = np.array(states)  # Shape: (num_windows, L*4)
    actions = np.array(actions).reshape(-1, 1)  # Shape: (num_windows, 1)
    dones = np.array(dones).reshape(-1, 1)

    return states, actions, scalers, dones


def get_dataset_from_results(
    res_object: dict,
    tps: TestPricingStrategy,
    n_days_to_use: Union[int, tuple[int, int]] = None,
    scalers: dict[str, Union[MinMaxScaler, dict[str, MinMaxScaler]]] = None,
):
    # Set up the cutoff point
    if n_days_to_use is not None:
        if isinstance(n_days_to_use, int):
            n_day_cutoff_start = 0
            n_day_cutoff_end = n_days_to_use * 96
        elif isinstance(n_days_to_use, tuple):
            n_day_cutoff_start = n_days_to_use[0] * 96
            n_day_cutoff_end = n_days_to_use[1] * 96
    else:
        n_day_cutoff_start = 0
        n_day_cutoff_end = None
    # Set up the common variables first
    t_amb = tps.t_amb[tps.start_idx : tps.stop_idx].to_numpy()[
        n_day_cutoff_start:n_day_cutoff_end
    ]
    agg_load = (
        np.array(res_object["res"]["net_load"])
        if "res" in res_object.keys()
        else np.array(res_object["net_load"])
    )[n_day_cutoff_start:n_day_cutoff_end]
    states, actions, dones = [], [], []
    # Scaler for t_amb
    if scalers is None:
        scalers = {"t_amb": MinMaxScaler(), "agg_load": MinMaxScaler()}
        t_amb_scaled = (
            scalers["t_amb"].fit_transform(downsample(t_amb).reshape(-1, 1)).flatten()
        )
    else:
        t_amb_scaled = (
            scalers["t_amb"].transform(downsample(t_amb).reshape(-1, 1)).flatten()
        )
    # Make the dataset for each consumer
    for icm, cm in enumerate(res_object["cons_models"]):
        pgrid_n = np.array(cm.net_load)[n_day_cutoff_start:n_day_cutoff_end]
        pgrid_not_n = agg_load - pgrid_n
        o_n = np.array(cm.th_mod.ac.override_history)[
            n_day_cutoff_start:n_day_cutoff_end
        ]
        a_t = np.array(cm.prices)[n_day_cutoff_start:n_day_cutoff_end]
        existing_scalers = scalers.get(f"cons_{icm}", None)
        states_n, actions_n, scalers_n, dones_n = prepare_replay_buffer(
            pgrid_n, pgrid_not_n, o_n, t_amb_scaled, a_t, L=8, scalers=existing_scalers
        )
        states.append(states_n)
        actions.append(actions_n)
        dones.append(dones_n)
        if existing_scalers is None:
            scalers[f"cons_{icm}"] = scalers_n
    return np.vstack(states), np.vstack(actions), scalers, np.vstack(dones)


def get_override_stats(cons_models: list[RealConsumerModel]):
    override_counts = np.zeros((len(cons_models), 96))
    for icm, cm in enumerate(cons_models):
        cons_overrides = (
            np.array(cm.th_mod.ac.override_history).reshape(-1, 96).sum(axis=0)
        )
        override_counts[icm] = cons_overrides.squeeze()
    return override_counts


class NormalizationEnv:
    def __init__(self, other: SingleConsTrainingEnvWithGlobalFullState):
        self.obs_space = other.observation_space
        self.act_space = other.action_space
        self.lg = other.lg
        self.tambg = other.tambg
        self.glg = other.glg
        self.n_prev_load = other.n_prev_load
        self.n_prev_tamb = other.n_prev_tamb

    def reset(self):
        self.prev_loads = [0.0] * self.n_prev_load
        self.prev_grid_loads = [0.0] * self.n_prev_load
        self.prev_tambs = [0.0] * self.n_prev_tamb
        self.prev_tins = [0.0] * self.n_prev_tamb
        self.prev_overrides = [0.0] * self.n_prev_tamb
        return [np.zeros_like(self.obs_space.sample())]

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
            [t / 24.0]
            + self.prev_loads
            + self.prev_grid_loads
            + self.prev_tambs
            + self.prev_tins
            + self.prev_overrides
        )
        return next_state


def gather_metrics_from_res(res: dict, load_cutoff: float = 200000.0):
    # Find the stuff
    net_load = res["net_load"] if "net_load" in res.keys() else res["res"]["net_load"]
    cons_models = res["cons_models"]
    # The load statistics are first
    peak_loads = net_load[net_load > load_cutoff].sum() / 4000_000.0
    tot_cons = net_load.sum() / 4000_000.0
    # The override stats are next
    ov_matrix = get_override_stats(cons_models)
    # n_slots = (
    #     np.array([cm.prices for cm in cons_models]) > 0.0
    # ).sum()
    n_slots = len(cons_models) * len(net_load)
    ov_frac = (ov_matrix.sum() / n_slots) if n_slots > 0 else 0.0
    return {
        "peak": peak_loads,
        "cons": tot_cons,
        "ov": ov_frac,
        "net_load": net_load,
    }


def show_heatmap(
    res_pack: dict,
    base_metrics: dict,
    key: Union[str, list[str]],
    title: Union[str, list[str]] = None,
    figsize: tuple = None,
):
    # collect the data
    run_idxs = np.array(list(res_pack.keys()))
    ips = np.unique(run_idxs[:, 0])
    ups = np.unique(run_idxs[:, 1])
    if isinstance(key, list):
        n_keys = len(key)
        fig, ax = plt.subplots(
            1, n_keys, figsize=(3 * n_keys, 4) if figsize is None else figsize
        )
    else:
        n_keys = 1
        key = [key]
        fig, _ax = plt.subplots(1, 1, figsize=(5, 5) if figsize is None else figsize)
        ax = [_ax]
        title = [title] if title is not None else title
    out_mat = np.zeros((n_keys, len(ips), len(ups)))
    for ind in range(n_keys):
        for xx, ip in enumerate(ips):
            for yy, up in enumerate(ups):
                this_res = res_pack[(ip, up)][key[ind]] / base_metrics[key[ind]]
                out_mat[ind][xx][yy] = this_res

        ax[ind].imshow(
            out_mat[ind],
            origin="lower",
            cmap="coolwarm",
            interpolation="none",
            vmin=0.1,
            vmax=2.5,
        )
        for xx, ip in enumerate(ips):
            for yy, up in enumerate(ups):
                ax[ind].text(
                    xx,
                    yy,
                    f"{out_mat[ind][xx][yy]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
        ax[ind].set(
            title=f"{key[ind]} comparison" if title is None else title[ind],
            ylabel="Update Frequency (days)",
            xlabel="Initial Period (days)",
        )
        ax[ind].set_xticks(range(len(ips)), labels=[str(l) for l in ips])
        ax[ind].set_yticks(range(len(ups)), labels=[str(l) for l in ups])
    fig.tight_layout()
    return ax


def get_plot_data(res, load_cutoff=200000.0):
    daily_cons = (res["net_load"].resample("1d").sum() / 400_000).to_numpy()
    dp_series = res["net_load"].copy()
    dp_series[dp_series < load_cutoff] = 0.0
    daily_peaks = (dp_series.resample("1d").sum() / 400_000).to_numpy()
    ovs = np.zeros_like(res["net_load"])
    for cm in res["cons_models"]:
        ovs += cm.th_mod.ac.override_history
    daily_ovs = ovs.reshape(-1, 96).sum(axis=1)
    return {
        "daily_cons": daily_cons,
        "daily_peaks": daily_peaks,
        "daily_ovs": daily_ovs,
    }


def make_plot_from_data(
    plot_data,
    base_plot_data,
    init_days,
    update_freq,
    smooth=False,
    cumulative=False,
    plot_overrides=True,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.set_title(
        f"Ratio of {'Cumulative' if cumulative else 'Daily'} Metrics Over Time"
    )
    ax.axhline(y=1.0, linestyle="--", c="k")
    ax.axvline(x=init_days - 1, linestyle="--", c="C3", alpha=0.5)
    for ix in range(init_days - 1, len(plot_data["daily_cons"]), update_freq):
        ax.axvline(x=ix, linestyle="--", c="C4", alpha=0.5)
    sm_f = lambda x: savgol_filter(x, 5, 2) if smooth else x
    cm_f = lambda x: np.cumsum(x) if cumulative else x
    ax.plot(
        sm_f(cm_f(plot_data["daily_cons"]) / cm_f(base_plot_data["daily_cons"])),
        label="Energy Consumption",
    )
    ax.plot(
        sm_f(cm_f(plot_data["daily_peaks"]) / cm_f(base_plot_data["daily_peaks"])),
        label="Energy at Peak Loads (>200 kW)",
    )
    if plot_overrides:
        ax.plot(
            sm_f(cm_f(plot_data["daily_ovs"]) / cm_f(base_plot_data["daily_ovs"])),
            label="Override Count",
        )
    ax.set(
        xlim=(0, len(plot_data["daily_cons"])),
        xlabel="Days",
        ylabel="Ratio (lower is better)",
    )
    ax.legend()
    fig.tight_layout()
