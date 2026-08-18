from .model_4r2c import ThermalModel4R2C
from .model_1r1c import ThermalModel1R1C
import numpy as np
import pandas as pd
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class SetpointFlexibilityModel:
    default_setpoint: float
    max_setpoint_deviation: float

    def __post_init__(self):
        self.ctrl_sig_to_setpoint = np.poly1d(
            np.polyfit(
                [-1, 1],
                [
                    self.default_setpoint - self.max_setpoint_deviation,
                    self.default_setpoint + self.max_setpoint_deviation,
                ],
                deg=1,
            )
        )
        a, b = self.ctrl_sig_to_setpoint.c
        self.setpoint_to_ctrl_sig = np.poly1d([1 / a, -b / a])


class RealConsumerModel:
    def __init__(
        self,
        load_profile: list,
        thermal_model: ThermalModel4R2C,
        setpoint_flexibility_model: SetpointFlexibilityModel,
    ) -> None:
        # The consumer's "tolerance" is embedded in the thermal model inside the ac specification
        self.load = load_profile
        self.th_mod = thermal_model
        self.spm = setpoint_flexibility_model
        self.t_ix = 0
        self.net_load = []
        self.prices = []

    def get_pd_series(self, index, return_prices=False):
        n_to_skip = max(len(self.net_load) - len(index), 0)
        if return_prices:
            return pd.Series(self.net_load[n_to_skip:], index=index), pd.Series(
                self.prices[n_to_skip:], index=index
            )
        return pd.Series(self.net_load[n_to_skip:], index=index)

    def reset(self, idx=None):
        self.t_ix = 0 if idx is None else idx
        self.net_load = []
        self.prices = []
        self.th_mod.reset()

    def step(self, pi: float, t_amb: float, q_irrad: float, elec_load: float = None):
        if elec_load is None:
            elec_load = self.load[self.t_ix % len(self.load)]
        p_ac = self.th_mod.step(
            self.spm.ctrl_sig_to_setpoint(pi), t_amb, q_irrad, elec_load
        )
        self.net_load.append(elec_load + p_ac)
        self.prices.append(pi)
        self.t_ix += 1
        return elec_load + p_ac


class PredictedConsumerModel:
    def __init__(
        self,
        load_template: list,
        thermal_model: ThermalModel1R1C,
        max_setpoint_deviation: float = 4.0,
    ) -> None:
        # This guy first takes historical data at a static price to fit the thermal model
        # Then it takes data with a predetermined price pattern to determine the consumer's price response
        self.load_template = load_template
        self.th_mod = thermal_model
        self.spm = SetpointFlexibilityModel(
            self.th_mod.ac.default_setpoint, max_setpoint_deviation
        )
        self.net_load = []
        self.prices = []
        self.t_ix = 0

    def reset(self, idx=None):
        self.t_ix = 0
        self.net_load = []
        self.prices = []
        self.th_mod.reset()

    def step(self, pi: float, t_amb: float, q_irrad: float, elec_load: float = None):
        if elec_load is None:
            elec_load = self.load_template[self.t_ix % len(self.load_template)]
        p_ac = self.th_mod.step(
            self.spm.ctrl_sig_to_setpoint(pi), t_amb, q_irrad, elec_load
        )
        self.t_ix += 1
        self.net_load.append(elec_load + p_ac)
        self.prices.append(pi)
        return elec_load + p_ac

    def copy_parameters_from(self, other):
        self.th_mod = deepcopy(other.th_mod)
        self.spm = deepcopy(other.spm)
