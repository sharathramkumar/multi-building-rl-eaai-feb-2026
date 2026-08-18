import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# A reward formulation that is based on P^2 after min-max scaling
def annotate_p2_rew_after_minmax(states: np.ndarray, L=8):
    # The index for the latest local load is 2L-1
    scaled_pg = states[:, 2 * L - 1]
    rewards = -((scaled_pg + 1) ** 2)
    return rewards


# A reward formulation that trades off energy term and override term
def annotate_p2_rew_after_minmax_with_ov_pen(
    states: np.ndarray, L=8, max_step_pen: float = 1.0
):
    # The index for the latest local load is 2L-1
    scaled_pg = states[:, 2 * L - 1]
    rewards = -((scaled_pg + 1) ** 2)
    # The index for the overrides in the previous window is 4L-1
    ov_frac = states[:, 4 * L - 1]
    ov_pen = ov_frac * max_step_pen
    rewards -= ov_pen
    return rewards


def calc_correlation(actual: np.ndarray, predic: np.ndarray):
    """
    Assumes actual and predic are np arrays of dimension (n_samples, n_features)
    """
    a_diff = actual - np.mean(actual, axis=0)
    p_diff = predic - np.mean(predic, axis=0)
    numerator = np.sum(a_diff * p_diff, axis=1)
    denominator = np.sqrt(np.sum(a_diff**2, axis=1)) * np.sqrt(
        np.sum(p_diff**2, axis=1)
    )
    return numerator / denominator


# A three term reward function for peak, consumption and overrides
def annotate_corr_p2_rew_after_minmax_with_ov_pen(
    states: np.ndarray,
    scalers: dict[str, dict[str, MinMaxScaler]],
    L=8,
    max_step_pen: float = 1.0,
    cons_ref_point: float = 0.7,
    n_cons: int = 100,
    return_components: bool = False,
    ignore_ov: bool = False,
    scaling_factors: dict = dict([("corr", 1.0), ("cons", 1.0), ("ov", 1.0)]),
):
    logger.debug(
        f"reward function received parameters: L={L}, max_step_pen={max_step_pen}, cons_ref_point={cons_ref_point}"
    )
    # Get the states for the correlation term
    f = np.poly1d(
        np.polyfit([-1, 0.5, 1], [0, 0.2, 3.5], deg=2)
    )  # An "equalizer" for the correlation
    scaled_pg = states[:, L : 2 * L]
    scaled_pg_n = states[:, 2 * L : 3 * L]
    tcorr = f(calc_correlation(scaled_pg, scaled_pg_n)) * scaling_factors["corr"]
    # The power consumption term
    n_samples_per_state = int(states.shape[0] / n_cons)
    tcons = np.zeros_like(tcorr)
    for cix in range(n_cons):
        this_ref_pg = (
            scalers[f"cons_{cix}"]["pg"].inverse_transform([[cons_ref_point]]).item()
        )
        this_cons_scaled_pg = states[
            cix * n_samples_per_state : (cix + 1) * n_samples_per_state, 2 * L - 1
        ]
        this_cons_act_pg = (
            scalers[f"cons_{cix}"]["pg"]
            .inverse_transform(this_cons_scaled_pg.reshape(-1, 1))
            .squeeze()
        )
        this_cons_score = this_cons_act_pg / this_ref_pg
        this_cons_score[
            this_cons_score < 0.2
        ] = -0.1  # -1.0 Give a boost to lower power consumption
        tcons[cix * n_samples_per_state : (cix + 1) * n_samples_per_state] = (
            this_cons_score * scaling_factors["cons"]
        )
    # The override term
    if not ignore_ov:
        ov_frac = states[:, 4 * L - 1]
        ov_pen = ov_frac * max_step_pen * scaling_factors["ov"]
    else:
        ov_pen = np.zeros_like(tcorr)
    # Total rew
    rew = -(tcorr + tcons + ov_pen)
    logger.info(
        f"mean reward components: corr={tcorr.mean()}, cons={tcons.mean()}, ov={ov_pen.mean()}, tot={rew.mean()}"
    )
    if return_components:
        return {"tcorr": tcorr, "tcons": tcons, "tov": ov_pen, "tot": rew}
    return rew


def annotate_ov_pen_only(
    states: np.ndarray,
    scalers=None,
    L: int = 8,
    max_step_pen: float = 1.0,
):
    # The override term
    ov_frac = states[:, 4 * L - 1]
    rew = -max_step_pen * ov_frac
    return rew
